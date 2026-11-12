// EPArena's one abstraction: Home + Window.
//
//   Home    producer-owned, symmetric, exact: the rows a rank produces, indexed by its
//           own tokens (x, top-k, expert output y). Peers address it via PeerTable.
//   Window  consumer-owned, bounded: the staging in front of a consumer (GEMM input,
//           ring, scratch, receive pool). The only buffer that is sized.
//   r       a row's compact index in the consumer's (key-major) receive list.
//
// The rule: row r is consumed in round r / W at slot r % W, and a slot is refilled
// only after its previous row was consumed. A row without a slot waits in its
// producer's Home, so nothing is dropped and nothing is sized for the worst case:
// routing only changes the number of rounds, which is bounded by max_rows / W.
//
// Libraries differ only in how their consumer frees a slot:
//   per row     overflow rows are read in place from Home (row() returns -1)
//               Mooncake (EPA-pull, tiered), DeepEP
//   per block   a per-slot credit counter (wait_free)                 Mega MoE ring
//   per launch  stream order; launch r covers clip(...) rows          dist_moe
//
// A Window may reserve `keyed` rows per key in front of the shared rows ("tier 1"):
// producers can then place a key's first rows without waiting for a reservation.
// With shared == 0 and keyed == the per-key maximum the window is the upstream
// static layout (EPA-pull's GEMM input).
//
// Host + device, header only. Includable from plain C++ (host sizers).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EPA_HD __host__ __device__ __forceinline__
#else
#define EPA_HD inline
#endif

namespace epa {

// Rows of a window that holds beta x the balanced load tokens * topk, rounded up.
EPA_HD int64_t window_rows(double beta, int64_t tokens, int64_t topk, int64_t align = 1) {
    const double rows = beta * static_cast<double>(tokens * topk);
    int64_t n = static_cast<int64_t>(rows);
    if (static_cast<double>(n) < rows) ++n;
    return (n + align - 1) / align * align;
}

// Turns an address in this rank's symmetric heap into the same offset in a peer's
// heap. Any provider works (cudaIpc tables, NCCL LSA windows, NVSHMEM, torch symm-mem).
struct PeerTable {
    void* local_base;            // this rank's heap base
    void* const* peer_bases;     // device array [num_ranks]; peer_bases[rank] == local_base

#if defined(__CUDACC__)
    template <typename T = void>
    __device__ __forceinline__ T* peer(const void* local_ptr, int rank) const {
        const auto off = reinterpret_cast<const char*>(local_ptr) - reinterpret_cast<const char*>(local_base);
        return reinterpret_cast<T*>(reinterpret_cast<char*>(peer_bases[rank]) + off);
    }
#endif
};

// A producer-owned region: row `row` of rank `rank`'s copy.
struct Home {
    PeerTable pt;
    const void* base;     // local address of the region (same offset on every rank)
    int64_t row_bytes;

#if defined(__CUDACC__)
    template <typename T = void>
    __device__ __forceinline__ T* row(int rank, int64_t row) const {
        return reinterpret_cast<T*>(pt.peer<char>(base, rank) + row * row_bytes);
    }
    template <typename T = void>
    __device__ __forceinline__ T* local_row(int64_t row) const {
        return reinterpret_cast<T*>(const_cast<char*>(static_cast<const char*>(base)) + row * row_bytes);
    }
#endif
};

struct Window {
    int64_t num_keys = 1;   // consumer keys (experts, (expert, source) pairs); 1 = unkeyed
    int64_t keyed = 0;      // rows per key placed without a reservation ("tier 1")
    int64_t shared = 0;     // rows shared by all keys: W
    int64_t max_rows = 0;   // bound on r for any routing (the old worst case): fixes the rounds

    // ------------------------------------------------------------------ sizing
    // A shared window for beta x the balanced load (unkeyed consumers).
    static EPA_HD Window sized(double beta, int64_t tokens, int64_t topk, int64_t max_rows,
                               int64_t align = 1) {
        Window w;
        w.shared = window_rows(beta, tokens, topk, align);
        if (w.shared > max_rows) w.shared = (max_rows + align - 1) / align * align;
        w.max_rows = max_rows;
        return w;
    }
    // Physical rows: the keyed prefix, then the shared rows.
    EPA_HD int64_t rows() const { return num_keys * keyed + shared; }

    // ------------------------------------------------------------------ the rule
    EPA_HD int64_t num_rounds() const {
        if (shared <= 0) return 1;
        const int64_t n = (max_rows + shared - 1) / shared;
        return n > 0 ? n : 1;
    }
    EPA_HD int64_t slot(int64_t r) const { return shared > 0 ? r % shared : r; }
    EPA_HD int64_t round(int64_t r) const { return shared > 0 ? r / shared : 0; }

    // Per-launch consumers: rows of the contiguous segment [begin, begin + count) of the
    // receive list that round `rnd` covers.
    EPA_HD int64_t clip(int64_t begin, int64_t count, int64_t rnd) const {
        const int64_t lo = rnd * shared, hi = lo + shared;
        const int64_t a = begin > lo ? begin : lo, b = begin + count < hi ? begin + count : hi;
        return b > a ? b - a : 0;
    }

    // Per-row consumers with a keyed prefix: rows of a key with n rows that need the
    // shared part, and whether a reservation [base, base + n) fits (-1: read from Home).
    EPA_HD int64_t overflow(int64_t n) const { return n > keyed ? n - keyed : 0; }
    EPA_HD int64_t accept(int64_t base, int64_t n) const { return base >= 0 and base + n <= shared ? base : -1; }

    // Physical row of the key's j-th row, given the key's shared base from accept();
    // -1 means the row stays in its producer's Home and is read in place.
    EPA_HD int64_t row(int64_t key, int64_t j, int64_t base = -1) const {
        if (j < keyed) return key * keyed + j;
        if (base < 0) return -1;
        return num_keys * keyed + base + (j - keyed);
    }

#if defined(__CUDACC__)
    // One fetch-add per (warp, destination): every lane asks for n_lane shared rows on
    // dst_lane's window, whose head counter is `head` (local address, translated with
    // pt). Returns this lane's accepted base, or -1 (read from Home). All 32 lanes call.
    __device__ __forceinline__ int64_t warp_reserve(const PeerTable& pt, int* head, int dst_lane, int n_lane) const {
        const unsigned full = 0xffffffffu;
        const int lane = threadIdx.x & 31;
        int64_t result = -1;
        unsigned pending = __ballot_sync(full, n_lane > 0);
        while (pending) {
            const int leader = __ffs(pending) - 1;
            const int dst = __shfl_sync(full, dst_lane, leader);
            const unsigned same = __ballot_sync(full, n_lane > 0 and dst_lane == dst) & pending;
            int prefix = 0, total = 0;
            for (unsigned m = same; m; m &= m - 1) {
                const int l = __ffs(m) - 1;
                const int v = __shfl_sync(full, n_lane, l);
                if (l < lane) prefix += v;
                total += v;
            }
            int base = 0;
            if (lane == leader) base = atomicAdd(pt.peer<int>(head, dst), total);
            base = __shfl_sync(full, base, leader);
            if (same >> lane & 1u) result = accept(base, total) >= 0 ? base + prefix : -1;
            pending &= ~same;
        }
        return result;
    }

    // Per-block consumers: wait until slot(r) was freed `releases` times per earlier
    // round (freed[] counts releases per slot).
    __device__ __forceinline__ void wait_free(const unsigned* freed, int64_t r, unsigned releases = 1) const {
        const unsigned need = static_cast<unsigned>(round(r)) * releases;
        while (*reinterpret_cast<const volatile unsigned*>(freed + slot(r)) < need) __nanosleep(32);
    }
#endif
};

}  // namespace epa
