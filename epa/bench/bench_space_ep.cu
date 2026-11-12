// epa::Space (EPArena v2) on a DeepEP-shaped MoE layer, R GPUs in one process.
// Build: nvcc -std=c++17 -O3 -arch=sm_100a -I ../include bench_space_ep.cu -o bench_space_ep
// Run:   ./bench_space_ep [zipf_s=1.5] [T=4096]        (prints JSON lines)
//
// One layer, as DeepEP runs it (expanded layout: one recv_x row per (token, expert)):
//   alloc  each GPU's notify block takes one recv_x page (and one y page unless leased) per
//          tile of P = 64 rows of a local expert
//   D      every GPU writes its tokens' rows into the destination's recv_x pages (NVLink)
//   E1     GEMM 1 stand-in: reads each x tile, writes h (2048 wide) to local scratch
//   E2     GEMM 2 stand-in: reads h, writes the y tile (7168 wide)
//   C      combine: reads every y row and writes it to its source GPU's [T][k] slots
//   R      reduce at the source: out[t] = sum over the k slots
//   free   gives the pages back (borrowed ones through the ReturnBuffer)
// The arithmetic does not depend on where pages live, so every configuration must give
// bitwise the same output as `upstream` (local pages only, enough of them).
#include <epa/space.cuh>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int H = 7168, I = 2048, K = 8, P = 64, E = 384;
constexpr int64_t kRowB = H * 2, kPageB = P * kRowB;
constexpr int kStageSlots = 32, kMoverBlocks = 32, kCombineSMs = 32;   // combine on few SMs, like DeepEP

// ------------------------------------------------------------------ kernels
__global__ void init_x_kernel(__nv_bfloat16* x, int64_t n, int seed) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        unsigned h = static_cast<unsigned>(i) * 2654435761u ^ (seed * 97u + 13u);
        h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
        x[i] = __float2bfloat16((static_cast<int>(h & 1023) - 512) / 256.0f);
    }
}

// alloc, in two passes with a barrier between them (every GPU first takes what it can
// locally, then the short ones borrow; otherwise a busy GPU drains a lender before the lender
// has taken its own pages). Codes [0, ntiles) are the y pages, [ntiles, 2 ntiles) the x pages,
// so local pages go to y first: a remote y page costs combine's reads, a remote x page only
// E1's. Naive baseline (no Board): one pass of per-thread first-fit takes.
__global__ void alloc_take_kernel(epa::Space sp, int n, int pass, int* codes, int* got, int* fail) {
    if (not sp.use_board) {
        if (pass == 1) return;
        for (int i = threadIdx.x; i < n; i += blockDim.x) { codes[i] = sp.take1(); if (codes[i] < 0) atomicAdd(fail, 1); }
        return;
    }
    if (pass == 0) {
        const int g0 = sp.block_take(n, codes, true, false);
        if (threadIdx.x == 0) *got = g0;
    } else {
        const int g0 = *got;
        const int g1 = g0 < n ? sp.block_take(n - g0, codes + g0, false, true) : 0;
        if (threadIdx.x == 0 and g0 + g1 < n) atomicAdd(fail, n - g0 - g1);
    }
}
__global__ void alloc_assign_kernel(epa::Space sp, epa::Object X, epa::Object Y, int ntiles, int sep_y, const int* codes,
                                    int* order_x, int* nlocal_x, int* order_y, int* nlocal_y, bool remote_first_x, bool remote_first_y) {
    auto pte = [](int c, unsigned st) { return c < 0 ? 0ull : epa::Pte::make(st, epa::ArenaGroup::owner_of_code(c), epa::ArenaGroup::page_of_code(c)); };
    for (int v = threadIdx.x; v < ntiles; v += blockDim.x) {
        if (sep_y) X.pte[v] = pte(codes[ntiles + v], epa::kActive), Y.pte[v] = pte(codes[v], epa::kIdle);
        else X.pte[v] = pte(codes[v], epa::kActive);
    }
    __syncthreads();
    // Tile order per consumer. Read in place: remote tiles first, so their NVLink latency
    // overlaps the other blocks' local tiles. Staged: local first, giving the Mover a head start.
    if (threadIdx.x == 0) {
        for (int pass = 0; pass < 2; ++pass) {
            const epa::Object& o = pass == 0 or not sep_y ? X : Y;
            const bool remote_first = pass == 0 ? remote_first_x : remote_first_y;
            int* ord = pass == 0 ? order_x : order_y;
            int k = 0;
            for (int side = 0; side < 2; ++side)
                for (int v = 0; v < ntiles; ++v)
                    if ((epa::Pte::owner(o.pte[v]) == sp.g.rank) == (side == (remote_first ? 1 : 0))) ord[k++] = v;
            int nl = 0;
            for (int v = 0; v < ntiles; ++v) nl += epa::Pte::owner(o.pte[v]) == sp.g.rank;
            *(pass == 0 ? nlocal_x : nlocal_y) = remote_first ? -1 : nl;   // -1: not staged
        }
    }
}

// D: warp per (token, expert) entry {t, dst, vtile, row}; dst's recv_x Ptes read over NVLink
__global__ void send_kernel(epa::Space sp, const int4* entries, int n, const __nv_bfloat16* x_in,
                            unsigned long long* const* x_pte_of) {
    const int lane = threadIdx.x & 31;
    for (int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32; w < n; w += gridDim.x * blockDim.x / 32) {
        const int4 e = entries[w];
        const unsigned long long pte = *reinterpret_cast<const volatile unsigned long long*>(x_pte_of[e.y] + e.z);
        if (epa::Pte::state(pte) == epa::kFree) continue;   // page not taken (counted as a failure)
        const int4* src = reinterpret_cast<const int4*>(x_in + int64_t(e.x) * H);
        int4* dst = reinterpret_cast<int4*>(sp.ptr(epa::Pte::code(pte)) + int64_t(e.w) * kRowB);
        #pragma unroll 4
        for (int i = lane; i < H / 8; i += 32) dst[i] = src[i];
    }
}

__device__ __forceinline__ void add8(float* acc, int4 v) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(b[k]); acc[2 * k] += f.x, acc[2 * k + 1] += f.y; }
}
__device__ __forceinline__ int4 pack8(const float* a) {
    int4 v;
    __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&v);
    #pragma unroll
    for (int k = 0; k < 4; ++k) b[k] = __floats2bfloat162_rn(a[2 * k], a[2 * k + 1]);
    return v;
}

// Source of tile `i` of an order: local page, remote page read in place, or staging slot
__device__ __forceinline__ const char* tile_src(const epa::Space& sp, const epa::Object& o, const int* order, int nlocal,
                                                int i, bool staged, const epa::Staging& st) {
    if (staged and nlocal >= 0 and i >= nlocal) { st.wait_ready(i - nlocal); return st.slot(i - nlocal); }
    return sp.ptr(epa::Pte::code(o.pte[order[i]]));
}

// E1: block per tile (in order); warp per row: h[c] = x[c] + x[c+2048] + x[c+4096] (+ x[c+6144])
__global__ void e1_kernel(epa::Space sp, epa::Object X, const int* order, const int* nlocal_p, int ntiles, int staged,
                          epa::Staging st, __nv_bfloat16* h) {
    const int nlocal = *nlocal_p, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    __shared__ const char* src_sh;
    for (int i = blockIdx.x; i < ntiles; i += gridDim.x) {
        if (threadIdx.x == 0) src_sh = tile_src(sp, X, order, nlocal, i, staged, st);
        __syncthreads();
        const int v = order[i];
        for (int r = warp; r < P; r += blockDim.x / 32) {
            const int4* xr = reinterpret_cast<const int4*>(src_sh + int64_t(r) * kRowB);
            int4* hr = reinterpret_cast<int4*>(h + (int64_t(v) * P + r) * I);
            for (int q = lane; q < I / 8; q += 32) {
                float acc[8] = {};
                add8(acc, xr[q]); add8(acc, xr[q + 256]); add8(acc, xr[q + 512]);
                if (q < 128) add8(acc, xr[q + 768]);
                hr[q] = pack8(acc);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 and staged and nlocal >= 0 and i >= nlocal) st.release(i - nlocal);
    }
}

// E2: block per tile; y[c] = h[c % 2048] * 0.5 + c / 2048  (y page local or remote, written in place)
__global__ void e2_kernel(epa::Space sp, epa::Object Y, int ntiles, const __nv_bfloat16* h) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    for (int v = blockIdx.x; v < ntiles; v += gridDim.x) {
        char* page = sp.ptr(epa::Pte::code(Y.pte[v]));
        for (int r = warp; r < P; r += blockDim.x / 32) {
            const int4* hr = reinterpret_cast<const int4*>(h + (int64_t(v) * P + r) * I);
            int4* yr = reinterpret_cast<int4*>(page + int64_t(r) * kRowB);
            for (int q = lane; q < H / 8; q += 32) {
                float acc[8] = {};
                add8(acc, hr[q % 256]);
                #pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] = acc[k] * 0.5f + static_cast<float>(q / 256);
                yr[q] = pack8(acc);
            }
        }
    }
}

// C: block per y tile (in order), warp per row -> its source GPU's slot [t][j]
__global__ void combine_kernel(epa::Space sp, epa::Object Y, const int* order, const int* nlocal_p, int ntiles, int staged,
                               epa::Staging st, const int4* meta, __nv_bfloat16* const* out_of) {
    const int nlocal = *nlocal_p, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    __shared__ const char* src_sh;
    for (int i = blockIdx.x; i < ntiles; i += gridDim.x) {
        if (threadIdx.x == 0) src_sh = tile_src(sp, Y, order, nlocal, i, staged, st);
        __syncthreads();
        const int v = order[i];
        for (int r = warp; r < P; r += blockDim.x / 32) {
            const int4 m = meta[int64_t(v) * P + r];   // {src, t, j, valid}
            if (m.w == 0) continue;
            const int4* yr = reinterpret_cast<const int4*>(src_sh + int64_t(r) * kRowB);
            int4* dst = reinterpret_cast<int4*>(out_of[m.x] + (int64_t(m.y) * K + m.z) * H);
            #pragma unroll 4
            for (int q = lane; q < H / 8; q += 32) dst[q] = yr[q];
        }
        __syncthreads();
        if (threadIdx.x == 0 and staged and nlocal >= 0 and i >= nlocal) st.release(i - nlocal);
    }
}

// The Mover's pull list: addresses of the remote tiles, in consumer order
__global__ void pull_list_kernel(epa::Space sp, epa::Object o, const int* order, const int* nlocal, char** pull, int* n_remote) {
    const int nl = *nlocal, n = o.num_vpages - nl;
    if (blockIdx.x == 0 and threadIdx.x == 0) *n_remote = n;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x)
        pull[j] = sp.ptr(epa::Pte::code(o.pte[order[nl + j]]));
}

__global__ void reduce_kernel(const __nv_bfloat16* out, __nv_bfloat16* fin, int T) {
    const int lane = threadIdx.x & 31;
    for (int t = (blockIdx.x * blockDim.x + threadIdx.x) / 32; t < T; t += gridDim.x * blockDim.x / 32)
        for (int q = lane; q < H / 8; q += 32) {
            float acc[8] = {};
            for (int j = 0; j < K; ++j) add8(acc, reinterpret_cast<const int4*>(out + (int64_t(t) * K + j) * H)[q]);
            reinterpret_cast<int4*>(fin + int64_t(t) * H)[q] = pack8(acc);
        }
}

__global__ void free_kernel(epa::Space sp, epa::Object X, epa::Object Y, int ntiles, int sep_y, int drain) {
    for (int v = threadIdx.x; v < ntiles; v += blockDim.x) {
        if (epa::Pte::state(X.pte[v]) != epa::kFree) sp.give(epa::Pte::code(X.pte[v]));
        if (sep_y and epa::Pte::state(Y.pte[v]) != epa::kFree) sp.give(epa::Pte::code(Y.pte[v]));
        X.pte[v] = 0, Y.pte[v] = 0;
    }
    __syncthreads();
    sp.flush(drain != 0);
}

// Eviction target pages: borrowed ones only (the point is to free local HBM)
// mode 0: local, then peers; 1: peers only (eviction); 2: local only (prefetch)
__global__ void take_codes_kernel(epa::Space sp, int* codes, int n, int mode, int* fail) {
    const int got = sp.block_take(n, codes, mode != 1, mode != 2);
    if (threadIdx.x == 0 and got < n) atomicAdd(fail, n - got);
}
__global__ void give_codes_kernel(epa::Space sp, const int* codes, int n, int flush = 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (codes[i] >= 0) sp.give(codes[i]);
    __syncthreads();
    if (flush) sp.flush(false);
}
// E1's use of the layer's weights: every page's first word must be there (local or not)
__global__ void touch_pages_kernel(epa::Space sp, epa::Object o, int salt, int* bad) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < o.num_vpages; i += gridDim.x * blockDim.x)
        if (*reinterpret_cast<const int*>(sp.ptr(epa::Pte::code(o.pte[i]))) != i * 7919 + salt) atomicAdd(bad, 1);
}
__global__ void set_ptes_kernel(epa::Object o, const int* codes, unsigned state) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < o.num_vpages; i += gridDim.x * blockDim.x)
        o.pte[i] = epa::Pte::make(state, epa::ArenaGroup::owner_of_code(codes[i]), epa::ArenaGroup::page_of_code(codes[i]));
}
__global__ void fill_pages_kernel(epa::Space sp, epa::Object o, int salt) {
    for (int i = blockIdx.x; i < o.num_vpages; i += gridDim.x) {
        int* p = reinterpret_cast<int*>(sp.ptr(epa::Pte::code(o.pte[i])));
        for (int k = threadIdx.x; k < kPageB / 4; k += blockDim.x) p[k] = i * 7919 + k + salt;
    }
}
__global__ void check_pages_kernel(epa::Space sp, epa::Object o, int salt, int* bad) {
    for (int i = blockIdx.x; i < o.num_vpages; i += gridDim.x) {
        const int* p = reinterpret_cast<const int*>(sp.ptr(epa::Pte::code(o.pte[i])));
        for (int k = threadIdx.x; k < kPageB / 4; k += blockDim.x) if (p[k] != i * 7919 + k + salt) { atomicAdd(bad, 1); break; }
    }
}
__global__ void diff_kernel(const int4* a, const int4* b, int64_t n, int* bad) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        if (a[i].x != b[i].x or a[i].y != b[i].y or a[i].z != b[i].z or a[i].w != b[i].w) atomicAdd(bad, 1);
}

// ------------------------------------------------------------------ host
struct Cfg { std::string name; double beta; int sep_y, use_board, staged, evict_pages, overlap = 0, extra = 0; };

int main(int argc, char** argv) {
    const double zipf_s = argc > 1 ? atof(argv[1]) : 1.5;
    const int T = argc > 2 ? atoi(argv[2]) : 4096;
    int R = 0;
    CHECK(cudaGetDeviceCount(&R));
    R = std::min(R, 8);
    const int L = E / R;
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        for (int p = 0; p < R; ++p) if (p != d) cudaDeviceEnablePeerAccess(p, 0);
    }
    cudaGetLastError();

    // Routing: Zipf(s) over a fixed permutation of experts, top-K by Gumbel noise per token
    std::mt19937 rng(1234);
    std::vector<int> perm(E);
    for (int e = 0; e < E; ++e) perm[e] = e;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> logp(E);
    for (int r = 0; r < E; ++r) logp[perm[r]] = -zipf_s * std::log(r + 1.0);
    std::uniform_real_distribution<double> U(1e-12, 1.0);
    std::vector<std::vector<int>> topk(R, std::vector<int>(size_t(T) * K));
    std::vector<int> cnt(E, 0);
    {
        std::vector<std::pair<double, int>> sc(E);
        for (int s = 0; s < R; ++s)
            for (int t = 0; t < T; ++t) {
                for (int e = 0; e < E; ++e) sc[e] = {logp[e] - std::log(-std::log(U(rng))), e};
                std::partial_sort(sc.begin(), sc.begin() + K, sc.end(), [](auto& a, auto& b) { return a.first > b.first; });
                for (int j = 0; j < K; ++j) topk[s][size_t(t) * K + j] = sc[j].second, ++cnt[sc[j].second];
            }
    }
    // Tiles per destination: experts' rows in tiles of P; entries and combine metadata
    std::vector<int> ntiles(R, 0), tile_base(E), row_ctr(E, 0);
    for (int e = 0; e < E; ++e) { const int g = e / L; tile_base[e] = ntiles[g]; ntiles[g] += (cnt[e] + P - 1) / P; }
    int max_tiles = 0, sum_tiles = 0;
    for (int g = 0; g < R; ++g) max_tiles = std::max(max_tiles, ntiles[g]), sum_tiles += ntiles[g];
    std::vector<std::vector<int4>> entries(R), meta(R);
    for (int g = 0; g < R; ++g) meta[g].assign(size_t(ntiles[g]) * P, int4{0, 0, 0, 0});
    for (int s = 0; s < R; ++s)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < K; ++j) {
                const int e = topk[s][size_t(t) * K + j], g = e / L, r = row_ctr[e]++;
                const int v = tile_base[e] + r / P, row = r % P;
                entries[s].push_back(int4{t, g, v, row});
                meta[g][size_t(v) * P + row] = int4{s, t, j, 1};
            }
    const double mean_tiles = double(sum_tiles) / R;
    printf("# R=%d T=%d K=%d E=%d zipf=%.2f tiles per GPU:", R, T, K, E, zipf_s);
    for (int g = 0; g < R; ++g) printf(" %d", ntiles[g]);
    printf(" (busiest %.2fx mean)\n", max_tiles / mean_tiles);

    // Per-GPU buffers that do not depend on the configuration
    struct Gpu {
        cudaStream_t s1, s2;
        __nv_bfloat16 *x_in, *out, *fin, *ref, *h;
        int4 *entries, *meta;
        unsigned long long *xpte, *ypte, *wpte;
        int *order_x, *order_y, *nl_x, *nl_y, *fail, *bad, *codes, *codes_old, *got, *codes2, *codes2_old, *fail2, *wbad;
        cudaEvent_t ev_alloc, ev_pref, ev_e2, ev_evict, ev_e1, ev_early;
        unsigned long long** xpte_of;
        __nv_bfloat16** out_of;
        epa::Staging st;
        char* region = nullptr;
        char** pull;
        int* n_remote;
        cudaEvent_t e0, e1;
    };
    std::vector<Gpu> G(R);
    std::vector<unsigned long long*> xpte_all(R);
    std::vector<__nv_bfloat16*> out_all(R);
    for (int g = 0; g < R; ++g) {
        Gpu& q = G[g];
        CHECK(cudaSetDevice(g));
        CHECK(cudaStreamCreateWithFlags(&q.s1, cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&q.s2, cudaStreamNonBlocking));
        CHECK(cudaMalloc(&q.x_in, size_t(T) * H * 2));
        CHECK(cudaMalloc(&q.out, size_t(T) * K * H * 2));
        CHECK(cudaMalloc(&q.fin, size_t(T) * H * 2));
        CHECK(cudaMalloc(&q.ref, size_t(T) * H * 2));
        CHECK(cudaMalloc(&q.h, size_t(max_tiles) * P * I * 2));
        CHECK(cudaMalloc(&q.entries, entries[g].size() * 16));
        CHECK(cudaMemcpy(q.entries, entries[g].data(), entries[g].size() * 16, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&q.meta, std::max<size_t>(1, meta[g].size()) * 16));
        CHECK(cudaMemcpy(q.meta, meta[g].data(), meta[g].size() * 16, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&q.xpte, max_tiles * 8)); CHECK(cudaMalloc(&q.ypte, max_tiles * 8)); CHECK(cudaMalloc(&q.wpte, 4096 * 8));
        CHECK(cudaMemset(q.xpte, 0, max_tiles * 8)); CHECK(cudaMemset(q.ypte, 0, max_tiles * 8));
        CHECK(cudaMalloc(&q.order_x, max_tiles * 4)); CHECK(cudaMalloc(&q.order_y, max_tiles * 4));
        CHECK(cudaMalloc(&q.nl_x, 4)); CHECK(cudaMalloc(&q.nl_y, 4));
        CHECK(cudaMalloc(&q.fail, 4)); CHECK(cudaMalloc(&q.bad, 4)); CHECK(cudaMalloc(&q.got, 4));
        CHECK(cudaMalloc(&q.codes, 4096 * 4)); CHECK(cudaMalloc(&q.codes_old, 4096 * 4));
        CHECK(cudaMalloc(&q.codes2, 4096 * 4)); CHECK(cudaMalloc(&q.codes2_old, 4096 * 4));
        CHECK(cudaMalloc(&q.fail2, 4)); CHECK(cudaMalloc(&q.wbad, 4));
        for (auto* e : {&q.ev_alloc, &q.ev_pref, &q.ev_e2, &q.ev_evict, &q.ev_e1, &q.ev_early}) CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        CHECK(cudaMalloc(&q.st.pages, kStageSlots * kPageB));
        CHECK(cudaMalloc(&q.st.ready, kStageSlots * 4)); CHECK(cudaMalloc(&q.st.done, kStageSlots * 4));
        q.st.num_slots = kStageSlots, q.st.page_bytes = kPageB;
        CHECK(cudaMalloc(&q.pull, max_tiles * 8)); CHECK(cudaMalloc(&q.n_remote, 4));
        CHECK(cudaEventCreateWithFlags(&q.e0, cudaEventDisableTiming)); CHECK(cudaEventCreateWithFlags(&q.e1, cudaEventDisableTiming));
        init_x_kernel<<<512, 256>>>(q.x_in, int64_t(T) * H, g);
        xpte_all[g] = q.xpte, out_all[g] = q.out;
    }
    for (int g = 0; g < R; ++g) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaMalloc(&G[g].xpte_of, R * 8)); CHECK(cudaMemcpy(G[g].xpte_of, xpte_all.data(), R * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&G[g].out_of, R * 8)); CHECK(cudaMemcpy(G[g].out_of, out_all.data(), R * 8, cudaMemcpyHostToDevice));
        CHECK(cudaDeviceSynchronize());
    }

    const int mean_t = static_cast<int>(std::ceil(mean_tiles));
    std::vector<Cfg> cfgs = {
        {"upstream",    -1.0, 1, 1, 0, 0},   // enough local pages for the busiest GPU
        {"naive",        1.0, 1, 0, 0, 0},   // pool 1.0x mean (x + y): first-fit peer pages, read in place
        {"board",        1.0, 1, 1, 0, 0},   // + Board, BorrowCache, ReturnBuffer
        {"board+stage",  1.0, 1, 1, 1, 0},   // + Mover: remote y tiles staged for combine (few SMs)
        {"lease",        1.0, 0, 1, 1, 0},   // + y leased in the recv_x page (no second page)
        {"lease@0.6",    0.6, 0, 1, 0, 0},   // smaller pool: lease, borrowing, no staging
        {"lease+stage@0.6", 0.6, 0, 1, 1, 0},
    };
    if (argc > 3 and std::string(argv[3]) == "evict")
        // Two layers' weights W0, W1 (256 pages = 235 MB each) alternate: layer i uses W(i % 2) in
        // E1; on a GPU where the layer does not fit next to both, the other one is parked on a peer.
        // seq: evict / prefetch as their own steps; overlap: on the Mover stream, prefetch during D
        // (E1 waits for it), evict after E2 during C / R / free (the next alloc waits for it)
        cfgs = {
            {"upstream", -1.0, 1, 1, 0, 0},
            {"lease@0.65, no weights", 0.65, 0, 1, 1, 0, 0, 512},
            {"lease@0.65 + weights, seq", 0.65, 0, 1, 1, 256, 0, 512},
            {"lease@0.65 + weights, overlap", 0.65, 0, 1, 1, 256, 1, 512},
            {"lease@0.65 + weights, overlap, evict after E1", 0.65, 0, 1, 1, 256, 2, 512},
            {"lease@0.65 + weights, overlap + early prefetch", 0.65, 0, 1, 1, 256, 3, 512},
        };
    const int iters = 8, warm = 2;
    const char* phase_names[] = {"alloc", "D", "E1", "E2", "C", "R", "free", "evict", "prefetch"};
    constexpr int NPH = 9;

    for (const Cfg& c : cfgs) {
        // Pool: same page count on every GPU
        const int N = c.beta < 0 ? 2 * max_tiles + 64 : static_cast<int>(c.beta * 2 * mean_t) + c.extra;
        epa::SpaceLayout lay{R, 1024, N};
        const int64_t arena_bytes = epa::Arena::bytes_for(N, kPageB);
        const int64_t region_bytes = epa::Arena::align_up(arena_bytes, 4096) + lay.bytes();
        std::vector<void*> bases(R);
        std::vector<void**> pt_dev(R);
        for (int g = 0; g < R; ++g) {
            CHECK(cudaSetDevice(g));
            if (G[g].region) CHECK(cudaFree(G[g].region));
            CHECK(cudaMalloc(&G[g].region, region_bytes));
            bases[g] = G[g].region;
        }
        int shift = 0;
        while ((N >> (shift + 1)) >= 32) ++shift;
        std::vector<epa::Space> sp(R);
        for (int g = 0; g < R; ++g) {
            CHECK(cudaSetDevice(g));
            const auto img = epa::Arena::host_init_image(N);
            CHECK(cudaMemcpy(G[g].region, img.data(), img.size(), cudaMemcpyHostToDevice));
            char* st = G[g].region + epa::Arena::align_up(arena_bytes, 4096);
            CHECK(cudaMemset(st, 0, lay.bytes()));
            std::vector<int> board(R, N >> shift);
            CHECK(cudaMemcpy(st + lay.board_off(), board.data(), R * 4, cudaMemcpyHostToDevice));
            CHECK(cudaMalloc(&pt_dev[g], R * 8));
            CHECK(cudaMemcpy(pt_dev[g], bases.data(), R * 8, cudaMemcpyHostToDevice));
            epa::Space& s = sp[g];
            s.g.local.base = G[g].region, s.g.local.page_bytes = kPageB, s.g.local.num_pages = N;
            s.g.pt = {bases[g], pt_dev[g]}, s.g.rank = g, s.g.num_ranks = R, s.g.peer_floor = 0;
            s.state = st, s.lay = lay, s.board_shift = shift, s.batch = 16, s.use_board = c.use_board;
        }
        auto obj = [&](unsigned long long* pte, int n, unsigned use, unsigned live) { epa::Object o; o.pte = pte, o.num_vpages = n, o.use_mask = use, o.live_mask = live; return o; };
        // Weights (eviction config): W pages per GPU, PARKED during the layer; filled once
        const int W = c.evict_pages;
        auto wobj = [&](int g, int k) { return obj(G[g].wpte + k * W, W, 0b00100 /* E1 */, ~0u); };
        const auto short_gpu = [&](int g) { return ntiles[g] * (c.sep_y ? 2 : 1) + 2 * W > N; };
        if (W > 0)
            for (int g = 0; g < R; ++g) {
                CHECK(cudaSetDevice(g));
                CHECK(cudaMemset(G[g].fail, 0, 4)); CHECK(cudaMemset(G[g].wbad, 0, 4)); CHECK(cudaMemset(G[g].fail2, 0, 4));
                for (int k = 0; k < 2; ++k) {
                    take_codes_kernel<<<1, 256>>>(sp[g], G[g].codes, W, 0, G[g].fail);
                    set_ptes_kernel<<<8, 256>>>(wobj(g, k), G[g].codes, epa::kParked);
                    fill_pages_kernel<<<64, 256>>>(sp[g], wobj(g, k), 2 * g + k);
                }
                CHECK(cudaDeviceSynchronize());
            }
        if (W > 0)   // start: both parked away where the layer does not fit next to them
            for (int g = 0; g < R; ++g) {
                if (not short_gpu(g)) continue;
                CHECK(cudaSetDevice(g));
                for (int k = 0; k < 2; ++k) {
                    take_codes_kernel<<<1, 256>>>(sp[g], G[g].codes, W, 1, G[g].fail);
                    epa::move_pages_kernel<<<64, 256>>>(sp[g], wobj(g, k), G[g].codes, G[g].codes_old);
                    give_codes_kernel<<<1, 256>>>(sp[g], G[g].codes_old, W);
                }
                CHECK(cudaDeviceSynchronize());
            }

        std::vector<std::vector<float>> ph(NPH);
        long long fails = 0, borrowed0 = 0;
        std::vector<cudaEvent_t> ev(R * (NPH + 1) * 2);
        for (int g = 0; g < R; ++g) { CHECK(cudaSetDevice(g)); for (int k = 0; k < (NPH + 1) * 2; ++k) CHECK(cudaEventCreate(&ev[g * (NPH + 1) * 2 + k])); }
        auto EV = [&](int g, int p, int k) { return ev[g * (NPH + 1) * 2 + p * 2 + k]; };

        std::vector<int> early(R, 0);   // pages of the next layer's weights already prefetched during E2
        auto wsub = [&](int g, int k, int off, int n) { return obj(G[g].wpte + k * W + off, n, 0b00100, ~0u); };
        for (int it = 0; it < warm + iters; ++it) {
            std::vector<float> t(NPH, 0.f);
            auto phase = [&](int p, auto launch) {
                for (int g = 0; g < R; ++g) {
                    CHECK(cudaSetDevice(g));
                    CHECK(cudaEventRecord(EV(g, p, 0), G[g].s1));
                    launch(g);
                    CHECK(cudaEventRecord(EV(g, p, 1), G[g].s1));
                }
                float mx = 0.f;
                for (int g = 0; g < R; ++g) {
                    CHECK(cudaSetDevice(g));
                    CHECK(cudaEventSynchronize(EV(g, p, 1)));
                    float ms;
                    CHECK(cudaEventElapsedTime(&ms, EV(g, p, 0), EV(g, p, 1)));
                    mx = std::max(mx, ms);
                }
                t[p] = mx * 1000.f;
            };
            // staged phase: the pull list is built on the device from the Ptes, then the Mover
            // (s2) and the consumer (s1) run together
            auto staged = [&](int g, const int* nl, const int* order, const epa::Object& o, auto consumer) {
                Gpu& q = G[g];
                if (not c.staged) { consumer(0); return; }
                CHECK(cudaMemsetAsync(q.st.ready, 0, kStageSlots * 4, q.s1));
                CHECK(cudaMemsetAsync(q.st.done, 0, kStageSlots * 4, q.s1));
                pull_list_kernel<<<8, 256, 0, q.s1>>>(sp[g], o, order, nl, q.pull, q.n_remote);
                CHECK(cudaEventRecord(q.e0, q.s1));
                CHECK(cudaStreamWaitEvent(q.s2, q.e0));
                epa::mover_pull_kernel<<<kMoverBlocks, 256, 0, q.s2>>>(q.st, q.pull, q.n_remote);
                CHECK(cudaEventRecord(q.e1, q.s2));
                consumer(1);
                CHECK(cudaStreamWaitEvent(q.s1, q.e1));
            };
            for (int g = 0; g < R; ++g) { CHECK(cudaSetDevice(g)); CHECK(cudaMemset(G[g].fail, 0, 4)); }
            const int wa = it % 2;   // this layer's weights
            // the copies, on stream `st` (s1: seq, s2: overlap)
            auto prefetch = [&](int g, cudaStream_t st) {   // back into local pages before E1 (what E2 of the last layer did not bring)
                Gpu& q = G[g];
                const int off = early[g], n = W - off;
                early[g] = 0;
                if (n <= 0) return;
                take_codes_kernel<<<1, 256, 0, st>>>(sp[g], q.codes2, n, 0, q.fail2);
                epa::move_pages_kernel<<<64, 256, 0, st>>>(sp[g], wsub(g, wa, off, n), q.codes2, q.codes2_old);
                give_codes_kernel<<<1, 256, 0, st>>>(sp[g], q.codes2_old, n, 1);
            };
            // early prefetch (overlap 3): during E2, which uses only local HBM, bring the next layer's
            // weights into the pages still free (pool - tiles - this layer's weights), local pages only
            auto early_prefetch = [&](int g, cudaStream_t st) {
                Gpu& q = G[g];
                const int k = std::max(0, std::min(W, N - ntiles[g] * (c.sep_y ? 2 : 1) - W - 8));
                early[g] = k;
                if (k == 0) return;
                take_codes_kernel<<<1, 256, 0, st>>>(sp[g], q.codes2, k, 2, q.fail2);
                epa::move_pages_kernel<<<64, 256, 0, st>>>(sp[g], wsub(g, 1 - wa, 0, k), q.codes2, q.codes2_old);
                give_codes_kernel<<<1, 256, 0, st>>>(sp[g], q.codes2_old, k, 1);
            };
            auto evict = [&](int g, cudaStream_t st) {      // out to peers after its last use (E2)
                Gpu& q = G[g];
                take_codes_kernel<<<1, 256, 0, st>>>(sp[g], q.codes2, W, 1, q.fail2);
                epa::move_pages_kernel<<<64, 256, 0, st>>>(sp[g], wobj(g, wa), q.codes2, q.codes2_old);
                give_codes_kernel<<<1, 256, 0, st>>>(sp[g], q.codes2_old, W, 0);   // local pages: no flush (s1 may be flushing)
            };
            const bool ov = c.overlap != 0;
            // evict after the weights' last use: E2 (overlap 1, seq) or E1 (overlap 2: during E2, which is local only)
            auto launch_evict = [&]() {
            if (W > 0) {
                    if (ov) {
                        for (int g = 0; g < R; ++g) {
                            if (not short_gpu(g)) continue;
                            CHECK(cudaSetDevice(g));
                            CHECK(cudaEventRecord(G[g].ev_e2, G[g].s1));
                            CHECK(cudaStreamWaitEvent(G[g].s2, G[g].ev_e2));
                            evict(g, G[g].s2);
                            CHECK(cudaEventRecord(G[g].ev_evict, G[g].s2));
                        }
                    } else {
                        phase(7, [&](int g) { if (short_gpu(g)) evict(g, G[g].s1); });
                    }
                }
            };
            phase(0, [&](int g) {   // pass 0: local pages only (waits for the previous layer's eviction)
                if (W > 0 and ov and short_gpu(g)) CHECK(cudaStreamWaitEvent(G[g].s1, G[g].ev_evict));
                alloc_take_kernel<<<1, 256, 0, G[g].s1>>>(sp[g], ntiles[g] * (c.sep_y ? 2 : 1), 0, G[g].codes, G[g].got, G[g].fail);
            });
            const float t_local = t[0];
            phase(0, [&](int g) {   // pass 1, after every GPU's pass 0: borrow the rest; Ptes and tile order
                Gpu& q = G[g];
                alloc_take_kernel<<<1, 256, 0, q.s1>>>(sp[g], ntiles[g] * (c.sep_y ? 2 : 1), 1, q.codes, q.got, q.fail);
                alloc_assign_kernel<<<1, 256, 0, q.s1>>>(sp[g], obj(q.xpte, ntiles[g], 0b00011, 0b00011), obj(q.ypte, ntiles[g], 0b01100, 0b01100),
                                                          ntiles[g], c.sep_y, q.codes, q.order_x, q.nl_x, q.order_y, q.nl_y, true, not c.staged);
            });
            t[0] += t_local;
            if (W > 0) {
                for (int g = 0; g < R; ++g) { CHECK(cudaSetDevice(g)); CHECK(cudaEventRecord(G[g].ev_alloc, G[g].s1)); }
                if (ov) {
                    for (int g = 0; g < R; ++g) {
                        if (not short_gpu(g)) continue;
                        CHECK(cudaSetDevice(g));
                        CHECK(cudaStreamWaitEvent(G[g].s2, G[g].ev_alloc));
                        prefetch(g, G[g].s2);
                        CHECK(cudaEventRecord(G[g].ev_pref, G[g].s2));
                    }
                } else {
                    phase(8, [&](int g) { if (short_gpu(g)) prefetch(g, G[g].s1); });
                }
            }
            phase(1, [&](int g) { send_kernel<<<264, 256, 0, G[g].s1>>>(sp[g], G[g].entries, int(entries[g].size()), G[g].x_in, G[g].xpte_of); });
            phase(2, [&](int g) {
                Gpu& q = G[g];
                if (W > 0) {   // E1 uses this layer's weights: wait for the prefetch, touch them
                    if (ov and short_gpu(g)) CHECK(cudaStreamWaitEvent(q.s1, q.ev_pref));
                    touch_pages_kernel<<<4, 256, 0, q.s1>>>(sp[g], wobj(g, wa), 2 * g + wa, q.wbad);
                }
                e1_kernel<<<148, 256, 0, q.s1>>>(sp[g], obj(q.xpte, ntiles[g], 0, 0), q.order_x, q.nl_x, ntiles[g], 0, q.st, q.h);   // E1 reads remote x from every SM: no staging
            });
            if (c.overlap == 2) launch_evict();
            if (W > 0 and c.overlap == 3)
                for (int g = 0; g < R; ++g) {
                    if (not short_gpu(g)) continue;
                    if (getenv("SPACE_DEBUG")) {
                        CHECK(cudaSetDevice(g)); CHECK(cudaDeviceSynchronize());
                        epa::ArenaMeta m; CHECK(cudaMemcpy(&m, sp[g].g.local.base, sizeof m, cudaMemcpyDeviceToHost));
                        printf("# it %d gpu %d before early: free %d\n", it, g, m.free_pages);
                    }
                    CHECK(cudaSetDevice(g));
                    CHECK(cudaEventRecord(G[g].ev_e1, G[g].s1));
                    CHECK(cudaStreamWaitEvent(G[g].s2, G[g].ev_e1));
                    early_prefetch(g, G[g].s2);
                    CHECK(cudaEventRecord(G[g].ev_early, G[g].s2));
                }
            phase(3, [&](int g) {
                Gpu& q = G[g];
                if (not c.sep_y) epa::lease_kernel<<<8, 256, 0, q.s1>>>(obj(q.ypte, ntiles[g], 0, 0), obj(q.xpte, ntiles[g], 0, 0), ntiles[g]);
                e2_kernel<<<148, 256, 0, q.s1>>>(sp[g], obj(q.ypte, ntiles[g], 0, 0), ntiles[g], q.h);
            });
            if (c.overlap != 2) launch_evict();
            phase(4, [&](int g) {
                Gpu& q = G[g];
                staged(g, q.nl_y, q.order_y, obj(q.ypte, ntiles[g], 0, 0), [&](int st) {
                    combine_kernel<<<kCombineSMs, 256, 0, q.s1>>>(sp[g], obj(q.ypte, ntiles[g], 0, 0), q.order_y, q.nl_y, ntiles[g], st, q.st, q.meta, q.out_of);
                });
            });
            phase(5, [&](int g) { reduce_kernel<<<264, 256, 0, G[g].s1>>>(G[g].out, G[g].fin, T); });
            phase(6, [&](int g) {
                Gpu& q = G[g];
                if (W > 0 and c.overlap == 3 and short_gpu(g)) CHECK(cudaStreamWaitEvent(q.s1, q.ev_early));   // both flush the ReturnBuffer
                free_kernel<<<1, 256, 0, q.s1>>>(sp[g], obj(q.xpte, ntiles[g], 0, 0), obj(q.ypte, ntiles[g], 0, 0), ntiles[g], c.sep_y, 0);
            });
            for (int g = 0; g < R; ++g) {
                int f;
                CHECK(cudaSetDevice(g));
                CHECK(cudaMemcpy(&f, G[g].fail, 4, cudaMemcpyDeviceToHost));
                if (it >= warm) fails += f;
            }
            if (it == warm - 1 and W > 0)
                for (int g = 0; g < R; ++g) { CHECK(cudaSetDevice(g)); CHECK(cudaDeviceSynchronize()); CHECK(cudaMemset(G[g].fail2, 0, 4)); }
            if (it == warm - 1)
                for (int g = 0; g < R; ++g) {
                    epa::ArenaMeta m;
                    CHECK(cudaSetDevice(g));
                    CHECK(cudaMemcpy(&m, sp[g].g.local.base, sizeof m, cudaMemcpyDeviceToHost));
                    borrowed0 += m.borrowed;
                }
            if (getenv("SPACE_DEBUG") and W > 0)
                for (int g = 0; g < R; ++g) {
                    if (not short_gpu(g)) continue;
                    CHECK(cudaSetDevice(g)); CHECK(cudaDeviceSynchronize());
                    int f2; epa::ArenaMeta m;
                    CHECK(cudaMemcpy(&f2, G[g].fail2, 4, cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(&m, sp[g].g.local.base, sizeof m, cudaMemcpyDeviceToHost));
                    printf("# it %d gpu %d fail2 %d free %d early %d\n", it, g, f2, m.free_pages, early[g]);
                }
            if (it >= warm) for (int p = 0; p < NPH; ++p) ph[p].push_back(t[p]);
        }
        // memory and borrowing
        long long borrowed = 0;
        int peak_pages = 0;
        for (int g = 0; g < R; ++g) {
            epa::ArenaMeta m;
            CHECK(cudaSetDevice(g));
            CHECK(cudaMemcpy(&m, sp[g].g.local.base, sizeof m, cudaMemcpyDeviceToHost));
            borrowed += m.borrowed;
            peak_pages = std::max(peak_pages, N - m.min_free_pages);
        }
        // correctness: bitwise against upstream; weights intact after every move
        int bad_total = 0;
        for (int g = 0; g < R; ++g) {
            CHECK(cudaSetDevice(g));
            if (c.name == "upstream") CHECK(cudaMemcpy(G[g].ref, G[g].fin, size_t(T) * H * 2, cudaMemcpyDeviceToDevice));
            CHECK(cudaMemset(G[g].bad, 0, 4));
            diff_kernel<<<264, 256>>>(reinterpret_cast<int4*>(G[g].fin), reinterpret_cast<int4*>(G[g].ref), int64_t(T) * H / 8, G[g].bad);
            CHECK(cudaStreamSynchronize(G[g].s2));
            if (W > 0) {
                for (int k = 0; k < 2; ++k) check_pages_kernel<<<64, 256>>>(sp[g], wobj(g, k), 2 * g + k, G[g].bad);
                int wb, f2;
                CHECK(cudaMemcpy(&wb, G[g].wbad, 4, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(&f2, G[g].fail2, 4, cudaMemcpyDeviceToHost));
                bad_total += wb, fails += f2;
            }
            int b;
            CHECK(cudaMemcpy(&b, G[g].bad, 4, cudaMemcpyDeviceToHost));
            bad_total += b;
            free_kernel<<<1, 256>>>(sp[g], obj(G[g].xpte, 0, 0, 0), obj(G[g].ypte, 0, 0, 0), 0, 0, 1);   // drain caches
            CHECK(cudaDeviceSynchronize());
        }
        auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.f : v[v.size() / 2]; };
        float layer = 0.f;
        printf("{\"cfg\": \"%s\", \"pool_pages\": %d, \"pool_MB\": %.1f, \"peak_local_pages\": %d, \"borrowed_pages_per_layer\": %.1f, "
               "\"failed_takes\": %lld, \"bitwise_equal\": %s",
               c.name.c_str(), N, N * kPageB / 1e6, peak_pages, double(borrowed - borrowed0) / iters, fails, bad_total == 0 ? "true" : "false");
        for (int p = 0; p < NPH; ++p) {
            const float m = med(ph[p]);
            if (p < 7 or W > 0) printf(", \"%s_us\": %.1f", phase_names[p], m);
            layer += m;
        }
        printf(", \"layer_us\": %.1f}\n", layer);
        fflush(stdout);
        for (auto& e : ev) cudaEventDestroy(e);
        for (int g = 0; g < R; ++g) { CHECK(cudaSetDevice(g)); CHECK(cudaFree(pt_dev[g])); }
    }
    return 0;
}
