// EPA-pull: Mooncake's low-latency dispatch/combine as a Home + Window instance
// (epa/window.cuh) with a per-row consumer and a single round.
//
//   Home    the producers' tensors below, read in place by peers (epa::Home)
//   Window  the GEMM input packed_recv_x itself: keys = local experts, keyed = R*T
//           rows per key, no shared rows, so every row has its slot in round 0 and
//           there is no symmetric receive buffer, capacity or reservation at all.
//
//   per rank (symmetric):  home_x [T][H]   (BF16, or FP8 + home_sf [T][H/128])
//                          home_topk [T][k]   int32
//                          back [T][k]        int32  row of (t, k) in the expert rank's y
//                          y [E/R][R*T][H]    BF16 expert output (written by the GEMM)
//                          flags [2][R], epochs
//
// dispatch  1. stage x (FP8 cast as upstream) and top-k into home; raise flag[me] on peers
//           2. wait all flags; count my entries per (local expert, source) over every
//              source's top-k (NVLink reads of k ints per token)
//           3. prefix -> layout_range / packed_recv_count exactly as upstream (rows of a
//              (expert, source) pair are contiguous)
//           4. pull rows, work split over (source, token) so a hot expert is spread over
//              the whole GPU; write src_info and the row id back to the source
// combine   1. raise flag[me]: my y is complete
//           2. wait all flags; for every token pull its k rows from the experts' ranks and
//              reduce in top-k order (same arithmetic as upstream)
//
// Reuse safety without ping-pong buffers: a rank rewrites home/y only after its next
// wait on every peer's flag, which every peer raises only after it finished pulling.
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <epa/epa.cuh>

namespace epa_pull {

struct Layout {
    // byte offsets in the symmetric heap
    int64_t flags, epochs, meta, home_x, home_sf, home_topk, back, y, total;

    static Layout make(int T, int H, int E, int K, int R) {
        auto al = [](int64_t x) { return (x + 255) / 256 * 256; };
        Layout l{};
        int64_t o = 0;
        l.flags = o, o = al(o + 2 * R * 4);
        l.epochs = o, o = al(o + 2 * 4);
        l.meta = o, o = al(o + 4);
        l.home_x = o, o = al(o + int64_t(T) * H * 2);          // BF16 size (FP8 uses half)
        l.home_sf = o, o = al(o + int64_t(T) * (H / 128) * 4);
        l.home_topk = o, o = al(o + int64_t(T) * K * 4);
        l.back = o, o = al(o + int64_t(T) * K * 4);
        l.y = o, o = al(o + int64_t(E / R) * R * T * H * 2);
        l.total = o;
        return l;
    }
    int64_t comm_bytes() const { return y; }   // everything but the expert output
};

struct Args {
    epa::PeerTable pt;
    epa::Window win;    // packed_recv_x: num_keys = E/R, keyed = R*T, shared = 0
    Layout l;
    const nv_bfloat16* x;
    const int64_t* topk_idx;
    const float* topk_weights;
    void* packed_recv_x;
    float* packed_recv_x_scales;
    int* packed_recv_src_info;
    int64_t* packed_recv_layout_range;
    int* packed_recv_count;
    nv_bfloat16* combined_x;
    int* work;          // local scratch: [L][R] counts, [L][R] cursors, [L][R] begins (zeroed)
    int num_tokens, T, E, K, R, rank;
};

template <typename T_ = void>
__device__ __forceinline__ T_* at(const Args& a, int64_t off) {
    return reinterpret_cast<T_*>(static_cast<char*>(a.pt.local_base) + off);
}

// The Home region at heap offset `off` with rows of `row_bytes`.
__device__ __forceinline__ epa::Home home(const Args& a, int64_t off, int64_t row_bytes) {
    return {a.pt, at<const void>(a, off), row_bytes};
}

__device__ __forceinline__ void st_release_sys(int* p, int v) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_sys(const int* p) {
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ int4 ld_nc(const int4* p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// Raise flag `which` on every peer (value = epoch), then wait until every peer has
// raised it; ends with a grid sync so all blocks see the peers' data.
__device__ __forceinline__ void flag_barrier(const Args& a, int which, int epoch) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();
    grid.sync();   // all local writes of this phase done
    int* flags = at<int>(a, a.l.flags) + which * a.R;
    if (blockIdx.x == 0 and threadIdx.x < a.R) {
        __threadfence_system();
        st_release_sys(a.pt.peer<int>(flags + a.rank, threadIdx.x), epoch);
        while (ld_acquire_sys(flags + threadIdx.x) < epoch) {}
    }
    grid.sync();
}

template <bool kFP8, int kHidden>
__global__ void __launch_bounds__(512, 1) dispatch(const Args a) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();
    constexpr int kScales = kHidden / 128;
    constexpr int kRowBytes = kFP8 ? kHidden : kHidden * 2;
    constexpr int kRowI4 = kRowBytes / 16;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
    const int gwarp = blockIdx.x * nwarps + warp, gnwarps = gridDim.x * nwarps;
    const int L = a.E / a.R, K = a.K, T = a.T;
    const int epoch = *at<volatile int>(a, a.l.epochs) + 1;
    int* cnt = a.work;                 // [L][R]
    int* cursor = a.work + L * a.R;    // [L][R]
    int* begin = a.work + 2 * L * a.R; // [L][R]

    // 1. stage x and top-k in home (same FP8 cast as upstream)
    auto* home_x = at<int4>(a, a.l.home_x);
    auto* home_sf = at<float>(a, a.l.home_sf);
    auto* home_topk = at<int>(a, a.l.home_topk);
    for (int t = gwarp; t < a.num_tokens; t += gnwarps) {
        const auto* src = reinterpret_cast<const int4*>(a.x) + int64_t(t) * (kHidden / 8);
        for (int i = lane; i < kHidden / 8; i += 32) {
            int4 v = __ldg(src + i);
            if constexpr (kFP8) {
                const auto* b = reinterpret_cast<const nv_bfloat16*>(&v);
                float f[8], amax = 1e-4f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = __bfloat162float(b[j]), amax = fmaxf(amax, fabsf(f[j]));
                #pragma unroll
                for (int m = 8; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
                const float scale = 448.0f / amax, scale_inv = amax * (1.0f / 448.0f);
                if (lane % 16 == 0) home_sf[int64_t(t) * kScales + i * 8 / 128] = scale_inv;
                int2 o;
                auto* o8 = reinterpret_cast<__nv_fp8x2_storage_t*>(&o);
                #pragma unroll
                for (int j = 0; j < 8; j += 2)
                    o8[j / 2] = __nv_cvt_float2_to_fp8x2(make_float2(f[j] * scale, f[j + 1] * scale), __NV_SATFINITE, __NV_E4M3);
                reinterpret_cast<int2*>(home_x)[int64_t(t) * (kHidden / 8) + i] = o;
            } else {
                home_x[int64_t(t) * kRowI4 + i] = v;
            }
        }
        if (lane < K) home_topk[t * K + lane] = static_cast<int>(__ldg(a.topk_idx + int64_t(t) * K + lane));
    }
    if (blockIdx.x == 0 and threadIdx.x == 0) *at<int>(a, a.l.meta) = a.num_tokens;
    flag_barrier(a, 0, epoch);
    const epa::Home h_meta = home(a, a.l.meta, 4), h_topk = home(a, a.l.home_topk, 4);
    const epa::Home h_x = home(a, a.l.home_x, kRowBytes), h_sf = home(a, a.l.home_sf, kScales * 4);
    const epa::Home h_back = home(a, a.l.back, 4);
    __shared__ int ntok[64];   // tokens staged by each source
    for (int s = threadIdx.x; s < a.R; s += blockDim.x)
        ntok[s] = *h_meta.row<const int>(s, 0);
    __syncthreads();

    // 2. count my entries per (local expert, source) over every source's top-k
    const int e_begin = a.rank * L;
    const int64_t n_entries = int64_t(a.R) * T * K;
    for (int64_t q = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; q < n_entries; q += int64_t(gridDim.x) * blockDim.x) {
        const int s = q / (int64_t(T) * K), tk = q % (int64_t(T) * K);
        if (tk / K >= ntok[s]) continue;
        const int e = *h_topk.row<const int>(s, tk) - e_begin;
        if (0 <= e and e < L) atomicAdd(cnt + e * a.R + s, 1);
    }
    grid.sync();

    // 3. prefix: rows of (expert, source) are contiguous, sources in order (as upstream)
    if (blockIdx.x == 0) {
        for (int e = threadIdx.x; e < L; e += blockDim.x) {
            int acc = 0;
            for (int s = 0; s < a.R; ++s) {
                const int c = cnt[e * a.R + s];
                begin[e * a.R + s] = acc;
                a.packed_recv_layout_range[e * a.R + s] =
                    (static_cast<int64_t>(acc) << 32) | static_cast<uint32_t>(c);   // pack2(count, begin)
                acc += c;
            }
            a.packed_recv_count[e] = acc;
        }
    }
    grid.sync();

    // 4. pull rows; one warp per 32 entries of (source, token, k), then row by row
    auto* rx = static_cast<int4*>(a.packed_recv_x);
    for (int64_t q0 = int64_t(gwarp) * 32; q0 < n_entries; q0 += int64_t(gnwarps) * 32) {
        const int64_t q = q0 + lane;
        int s = -1, t = 0, k = 0, e = -1;
        if (q < n_entries) {
            s = q / (int64_t(T) * K);
            const int tk = q % (int64_t(T) * K);
            t = tk / K, k = tk % K;
            if (t < ntok[s])
                e = *h_topk.row<const int>(s, tk) - e_begin;
        }
        const bool mine = 0 <= e and e < L;
        int pos = 0;
        if (mine) {
            pos = begin[e * a.R + s] + atomicAdd(cursor + e * a.R + s, 1);
            const int64_t row = a.win.row(e, pos);
            a.packed_recv_src_info[row] = (k << 24) | t;
            *h_back.row<int>(s, t * K + k) = static_cast<int>(row);
        }
        unsigned m = __ballot_sync(0xffffffff, mine);
        while (m) {
            const int l = __ffs(m) - 1;
            m &= m - 1;
            const int ss = __shfl_sync(0xffffffff, s, l), tt = __shfl_sync(0xffffffff, t, l);
            const int ee = __shfl_sync(0xffffffff, e, l), pp = __shfl_sync(0xffffffff, pos, l);
            const int64_t row = a.win.row(ee, pp);
            const int4* src = h_x.row<const int4>(ss, tt);
            int4* dst = rx + row * kRowI4;
            constexpr int U = 7;
            int i = lane;
            for (; i + 32 * (U - 1) < kRowI4; i += 32 * U) {
                int4 v[U];
                #pragma unroll
                for (int u = 0; u < U; ++u) v[u] = ld_nc(src + i + 32 * u);
                #pragma unroll
                for (int u = 0; u < U; ++u) dst[i + 32 * u] = v[u];
            }
            for (; i < kRowI4; i += 32) dst[i] = ld_nc(src + i);
            if constexpr (kFP8) {
                const float* ssf = h_sf.row<const float>(ss, tt);
                float* dsf = a.packed_recv_x_scales + int64_t(ee) * a.win.keyed * kScales + pp;
                for (int j = lane; j < kScales; j += 32) dsf[int64_t(j) * a.R * T] = ssf[j];
            }
        }
    }
    grid.sync();
    // reset scratch and advance the epoch for the next call
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 3 * L * a.R; i += gridDim.x * blockDim.x) a.work[i] = 0;
    if (blockIdx.x == 0 and threadIdx.x == 0) *at<int>(a, a.l.epochs) = epoch;
}

template <int kHidden>
__global__ void __launch_bounds__(512, 1) combine(const Args a) {
    constexpr int kRowI4 = kHidden * 2 / 16;
    constexpr int kMaxTopk = 8;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
    const int gwarp = blockIdx.x * nwarps + warp, gnwarps = gridDim.x * nwarps;
    const int L = a.E / a.R, K = a.K;
    const int epoch = *at<volatile int>(a, a.l.epochs + 4) + 1;

    // 1. my y is complete (previous kernels on this stream); 2. wait for every rank's
    flag_barrier(a, 1, epoch);

    // Work item = (token, chunk of 64 int4): 14 items per token keep every warp busy
    // even at T = 128, and each lane has 2 * k independent NVLink loads in flight.
    constexpr int U = 2, kChunk = 32 * U, kChunks = (kRowI4 + kChunk - 1) / kChunk;
    const epa::Home h_y = home(a, a.l.y, kRowI4 * 16);
    const auto* back = at<const int>(a, a.l.back);
    for (int item = gwarp; item < a.num_tokens * kChunks; item += gnwarps) {
        const int t = item / kChunks, c = item % kChunks;
        // lane k < K fetches the row address and weight of top-k entry k
        const int4* my_row = nullptr;
        float my_w = 0;
        if (lane < K) {
            const int e = static_cast<int>(__ldg(a.topk_idx + int64_t(t) * K + lane));
            if (e >= 0) {
                my_row = h_y.row<const int4>(e / L, back[t * K + lane]);
                my_w = __ldg(a.topk_weights + int64_t(t) * K + lane);
            }
        }
        const int4* rows[kMaxTopk];
        float w[kMaxTopk];
        #pragma unroll
        for (int k = 0; k < kMaxTopk; ++k) {
            rows[k] = reinterpret_cast<const int4*>(__shfl_sync(0xffffffff, reinterpret_cast<uintptr_t>(my_row), k));
            w[k] = __shfl_sync(0xffffffff, my_w, k);
        }
        int4 v[U][kMaxTopk];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = c * kChunk + u * 32 + lane;
            #pragma unroll
            for (int k = 0; k < kMaxTopk; ++k)
                if (k < K and rows[k] != nullptr and i < kRowI4) v[u][k] = ld_nc(rows[k] + i);
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = c * kChunk + u * 32 + lane;
            if (i >= kRowI4) continue;
            float acc[8] = {};
            #pragma unroll
            for (int k = 0; k < kMaxTopk; ++k) {
                if (k < K and rows[k] != nullptr) {
                    const auto* b = reinterpret_cast<const nv_bfloat16*>(&v[u][k]);
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) acc[q] += __bfloat162float(b[q]) * w[k];
                }
            }
            int4 o;
            auto* ob = reinterpret_cast<nv_bfloat16*>(&o);
            #pragma unroll
            for (int q = 0; q < 8; ++q) ob[q] = __float2bfloat16(acc[q]);
            reinterpret_cast<int4*>(a.combined_x)[int64_t(t) * kRowI4 + i] = o;
        }
    }
    cooperative_groups::this_grid().sync();
    if (blockIdx.x == 0 and threadIdx.x == 0) *at<int>(a, a.l.epochs + 4) = epoch;
}

}  // namespace epa_pull
