#pragma once

// Paged expanded output (do_expand=True): recv_x and the expert output y without the
// worst-case [R * T * min(k, E_local)] rows.
//
// recv_x is the data region of a local epa::Arena of N pages of P rows; y and the
// expanded weights are ordinary tensors with the same row numbering. A page holds P rows
// of one local expert, so with P = the grouped GEMM's M alignment a page is one tile:
// tile_expert[page] is the expert of that tile, or -1 if the page is free (the
// contiguous-layout grouped GEMM's m_indices, one entry per tile instead of per row).
//
// Pages are sticky per local expert: expert e keeps held[e] pages across calls in its own
// table etable[e][0, held). Per call (dispatch kernel, SM 0, once the expert counts are
// known), each expert e with need = ceil(n_e / P):
//   trims to need + slack if it holds more than need + 2 x slack (pages given back)
//   grows to need + slack if it holds fewer than need (else need, after a full trim)
//   marks pages [0, need) valid (tile_expert = e) and [need, held) free (-1), touching
//   only the pages whose state changed
// With steady counts (decode) no page moves and nothing is written. The previous call's
// expert and combine have finished by then (stream order), so pages can move.
// The copy epilogue puts row r of expert e at physical row
//   etable[e][r / P] * P + r % P            (r / P < avail[e], else dropped and counted)
// and records that index in recv_src_metadata, so combine reads y unchanged.
//
// Meta block (ints), L = local experts, N = pages:
//   [0] pages held after the last call  [1] failed takes   [2] dropped rows
//   [3] most pages held by one call     [4] trim/retry flag [5..15] spare
//   cnt[L] row counters | avail[L + 1] valid pages per expert | held[L] | etable[L][N] | tile_expert[N]
//   | tiles[N + 1]: tiles[0] = valid tiles this call, tiles[1 + t] = page of valid tile t, grouped by
//   expert (a dense tile list for a grouped GEMM, like upstream's contiguous tiles)
//   | tiles_y[N + 1] | etable_y[L][N]: the y page paired with each recv_x page (the same page unless
//   the pool is shared with the EP buffer, layout/ep/epa.cuh)
#include <epa/arena.cuh>

namespace deep_ep::layout {

struct EPAOutLayout {
    int* meta;
    int num_local_experts, num_pages, rows_per_page;

    static constexpr int kHeaderInts = 16;
    __host__ __device__ static int64_t num_meta_ints(int L, int N) {
        return kHeaderInts + L + (L + 1) + L + static_cast<int64_t>(L) * N + N + 2 * (N + 1) + static_cast<int64_t>(L) * N + L + 1 + N;
    }

    __host__ __device__ int* held() const { return meta + 0; }
    __host__ __device__ int* failed() const { return meta + 1; }
    __host__ __device__ int* dropped_rows() const { return meta + 2; }
    __host__ __device__ int* max_held() const { return meta + 3; }
    __host__ __device__ int* flag() const { return meta + 4; }
    __host__ __device__ int* cnt() const { return meta + kHeaderInts; }
    __host__ __device__ int* avail() const { return cnt() + num_local_experts; }
    __host__ __device__ int* held_e() const { return avail() + num_local_experts + 1; }
    __host__ __device__ int* etable() const { return held_e() + num_local_experts; }
    __host__ __device__ int* tile_expert() const { return etable() + static_cast<int64_t>(num_local_experts) * num_pages; }
    __host__ __device__ int* tiles() const { return tile_expert() + num_pages; }
    // y page of each tile: the recv_x page itself, or (shared pool) a zone-B page paired with it
    __host__ __device__ int* tiles_y() const { return tiles() + num_pages + 1; }
    __host__ __device__ int* etable_y() const { return tiles_y() + num_pages + 1; }
    // Local (own-pool) valid tiles per key; avail counts borrowed peer tiles too
    __host__ __device__ int* lvalid() const { return etable_y() + static_cast<int64_t>(num_local_experts) * num_pages; }
    // Peer spill: header [11] pages borrowed (stat), [12] this GPU's first global page id in the
    // window view, [13] pages per GPU in the window (0: spill off). Table entries < 0 are borrowed
    // peer pages, -(2 + global id); entries >= 0 are own pages
    // Peer spill: own pages of this call's partly borrowed tiles, given back at the next call
    __host__ __device__ int* tmp_n() const { return lvalid() + num_local_experts; }
    __host__ __device__ int* tmp() const { return tmp_n() + 1; }
    __host__ __device__ int gid(const int& v) const { return v >= 0 ? meta[12] + v : -2 - v; }

    __host__ __device__ epa::Arena arena(char* base, int64_t row_bytes) const {
        epa::Arena a;
        a.base = base;
        a.page_bytes = row_bytes * rows_per_page;
        a.num_pages = num_pages;
        return a;
    }
};

// The output arena is used by one GPU only, so its ring needs GPU-scope ordering, not the
// system scope epa::Arena uses for peers (a system-scope release costs ~1.5 us, and the first
// one on an SM after launch far more). Same ring protocol as epa::Arena.
#if defined(__CUDACC__)
namespace epa_out {
template <typename T> using dev_ref = cuda::atomic_ref<T, cuda::thread_scope_device>;

// Return the n pages of `pages` with ticket t0 = tail fetch-add (by one thread); slots i of
// [0, n) filled by any thread.
__device__ __forceinline__ void give_slot(const epa::Arena& a, unsigned long long t0, int i, int page) {
    const unsigned long long t = t0 + i;
    dev_ref<unsigned long long> s(a.seq()[t % a.num_pages]);
    while ((s.load(cuda::memory_order_acquire) >> epa::Arena::kIdBits) != epa::Arena::seq_of(t)) __nanosleep(20);
    s.store(((t + 1) << epa::Arena::kIdBits) | static_cast<unsigned>(page), cuda::memory_order_release);
}
// n pages or -1 (one thread).
// floor: pages that must stay free (peer spill keeps them for receive-page growth).
__device__ __forceinline__ long long try_take(const epa::Arena& a, int n, int floor = 0) {
    dev_ref<int> f(a.meta()->free_pages);
    int cur = f.load(cuda::memory_order_relaxed);
    while (cur >= n + floor and not f.compare_exchange_weak(cur, cur - n, cuda::memory_order_relaxed)) {}
    if (cur < n + floor)
        return -1;
    return static_cast<long long>(dev_ref<unsigned long long>(a.meta()->head).fetch_add(n, cuda::memory_order_relaxed));
}
__device__ __forceinline__ int page_at(const epa::Arena& a, unsigned long long ticket, int i) {
    const unsigned long long t = ticket + i;
    dev_ref<unsigned long long> s(a.seq()[t % a.num_pages]);
    unsigned long long w;
    while (((w = s.load(cuda::memory_order_acquire)) >> epa::Arena::kIdBits) != epa::Arena::seq_of(t + 1)) __nanosleep(20);
    s.store((t + a.num_pages) << epa::Arena::kIdBits, cuda::memory_order_release);
    return static_cast<int>(w & ((1u << epa::Arena::kIdBits) - 1));
}
}  // namespace epa_out
#endif

}  // namespace deep_ep::layout
