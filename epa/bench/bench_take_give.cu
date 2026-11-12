// Cost of epa::Arena page take / give, timed on the GPU.
// Build: nvcc -std=c++17 -O3 -arch=sm_100a -I ../include bench_take_give.cu -o bench_take_give
// Run:   ./bench_take_give <dir with gen_take_patterns.py output>   (prints JSON lines)
//
//   ops       one thread, arena on the same GPU or on a peer over NVLink: try_reserve,
//             take_reserved, page_at, give; and the warp forms (page_at by 32 lanes, warp_give)
//   conc      S SMs x W warps, each warp's lane 0 loops take(1) + give(1) on one arena
//             (local, a peer's, or all 8 GPUs on GPU 0's)
//   deepep    per call: one thread per GPU takes want[d] pages of each of the 8 arenas,
//             a block writes the page tables; then each receiver warp_gives per source
//   mooncake  per call: 48 SMs x 1 count warp per GPU, lane = expert, warp_try_take
//             grouped by destination; then each receiver warp_gives per (expert, source)
//   distmoe   one page per round: a 1-thread kernel wait_take(1) + page_at, a 1-thread
//             kernel give(1), local, replayed in a CUDA graph
//   mega      modeled (MegaMoE's ring uses Window credits, not the arena): every SM takes
//             one local page per 128-row block and gives it back after the GEMM block
#include <epa/arena.cuh>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int R = 8, E = 384, kMaxPages = 256;

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void wait_go(const volatile int* go) { while (*go == 0) __nanosleep(100); }

struct Arenas { epa::Arena a[R]; };

// Arena::warp_give with the full-warp mask. The library's version takes __activemask(),
// which can be a strict subset of the warp after divergent code (seen here: fffffffe +
// 00000001); each subset then does its own tail fetch-add and waits for slots no take
// will fill, so the kernel hangs. Same body otherwise.
__device__ __forceinline__ void warp_give_full(const epa::Arena& a, const int* pages, int n) {
    using U = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
    __syncwarp();
    const int lane = threadIdx.x & 31;
    unsigned long long t0 = 0;
    if (lane == 0) t0 = U(a.meta()->tail).fetch_add(n, cuda::memory_order_relaxed);
    t0 = __shfl_sync(0xffffffffu, t0, 0);
    for (int i = lane; i < n; i += 32) {
        const unsigned long long t = t0 + i;
        U s(a.seq()[static_cast<int>(t % a.num_pages)]);
        while ((s.load(cuda::memory_order_acquire) >> epa::Arena::kIdBits) != epa::Arena::seq_of(t)) __nanosleep(20);
        s.store(((t + 1) << epa::Arena::kIdBits) | static_cast<unsigned>(pages[i]), cuda::memory_order_release);
    }
    __syncwarp();
    if (lane == 0) cuda::atomic_ref<int, cuda::thread_scope_system>(a.meta()->free_pages).fetch_add(n, cuda::memory_order_release);
}

// ------------------------------------------------------------------ ops (one warp)
// out per iter: reserve, take_reserved, page_at x n (one thread), give n (one thread),
// page_at x n (32 lanes), warp_give n; clk = {clock, gtime} at start and end
__global__ void ops_kernel(epa::Arena a, int n, int iters, long long* out, long long* clk) {
    __shared__ int pages[kMaxPages];
    const int lane = threadIdx.x;
    if (lane == 0) clk[0] = clock64(), clk[1] = gtime();
    for (int it = 0; it < iters; ++it) {
        long long* o = out + it * 6;
        if (lane == 0) {
            const long long t0 = clock64();
            const bool ok = a.try_reserve(n);
            const long long t1 = clock64();
            const unsigned long long tk = ok ? a.take_reserved(n) : 0;
            const long long t2 = clock64();
            for (int i = 0; ok and i < n; ++i) pages[i] = a.page_at(tk, i);
            const long long t3 = clock64();
            if (ok) a.give(pages, n);
            const long long t4 = clock64();
            o[0] = ok ? t1 - t0 : -1, o[1] = t2 - t1, o[2] = t3 - t2, o[3] = t4 - t3;
        }
        __syncwarp();
        long long tk = 0;
        if (lane == 0) tk = a.try_take(n);
        tk = __shfl_sync(0xffffffffu, tk, 0);
        __syncwarp();
        const long long t5 = clock64();
        for (int i = lane; tk >= 0 and i < n; i += 32) pages[i] = a.page_at(static_cast<unsigned long long>(tk), i);
        __syncwarp();
        const long long t6 = clock64();
        if (tk >= 0) warp_give_full(a, pages, n);
        __syncwarp();
        const long long t7 = clock64();
        if (lane == 0) o[4] = tk >= 0 ? t6 - t5 : -1, o[5] = t7 - t6;
        __syncwarp();
    }
    if (lane == 0) clk[2] = clock64(), clk[3] = gtime();
}

// ------------------------------------------------------------------ conc
__global__ void conc_kernel(epa::Arena a, int iters, long long* lat, unsigned long long* span, const volatile int* go,
                            long long* clk) {
    const int lane = threadIdx.x & 31;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (lane != 0) return;
    wait_go(go);
    if (warp == 0) clk[0] = clock64(), clk[1] = gtime();
    const unsigned long long s = gtime();
    for (int i = 0; i < iters; ++i) {
        const long long t0 = clock64();
        const long long tk = a.try_take(1);
        const int p = tk < 0 ? -1 : a.page_at(static_cast<unsigned long long>(tk), 0);
        const long long t1 = clock64();
        if (p >= 0) a.give(p);
        const long long t2 = clock64();
        lat[(warp * iters + i) * 2] = p < 0 ? -1 : t1 - t0;
        lat[(warp * iters + i) * 2 + 1] = p < 0 ? -1 : t2 - t1;
    }
    span[warp * 2] = s, span[warp * 2 + 1] = gtime();
    if (warp == 0) clk[2] = clock64(), clk[3] = gtime();
}

// ------------------------------------------------------------------ deepep
// times: [0] take atomics done, [1] tables written (ns from start)
__global__ void deepep_take(Arenas as, const int* want, int* got, int* ltable, int* const* rtab,
                            unsigned long long* times, const volatile int* go) {
    __shared__ long long tk[R];
    __shared__ int pre[R + 1];
    __shared__ unsigned long long t0;
    if (threadIdx.x == 0) {
        wait_go(go);
        t0 = gtime();
        pre[0] = 0;
        for (int d = 0; d < R; ++d) {
            tk[d] = want[d] > 0 ? as.a[d].try_take(want[d]) : -1;
            got[d] = tk[d] >= 0 ? want[d] : 0;
            pre[d + 1] = pre[d] + got[d];
        }
        times[0] = gtime() - t0;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < pre[R]; q += blockDim.x) {
        int d = 0;
        while (q >= pre[d + 1]) ++d;
        const int i = q - pre[d];
        const int id = as.a[d].page_at(static_cast<unsigned long long>(tk[d]), i);
        ltable[d * kMaxPages + i] = id;
        rtab[d][i] = id;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) times[1] = gtime() - t0;
}
// receiver: warp s gives back the pages source s took (its own arena, local)
__global__ void deepep_give(epa::Arena a, const int* rtable, const int* taken, unsigned long long* times,
                            const volatile int* go) {
    __shared__ unsigned long long t0;
    if (threadIdx.x == 0) wait_go(go), t0 = gtime();
    __syncthreads();
    const int w = threadIdx.x / 32;
    if (w < R and taken[w] > 0) warp_give_full(a, rtable + w * kMaxPages, taken[w]);
    __syncthreads();
    if (threadIdx.x == 0) times[0] = gtime() - t0;
}

// ------------------------------------------------------------------ mooncake
// 48 blocks x 32 threads; lane l < 8 of block b owns expert b * 8 + l (EPA_TAKE_PAGES)
__global__ void mc_take(Arenas as, int src, const int* want, int* got, int* ltable, int* const* rtab_base,
                        unsigned long long* span, const volatile int* go) {
    const int lane = threadIdx.x;
    if (lane == 0) wait_go(go);
    __syncwarp();
    const unsigned long long s = gtime();
    const bool owns = lane < 8;
    const int i = blockIdx.x * 8 + lane;
    const int dst = owns ? i / (E / R) : 0, el = owns ? i % (E / R) : 0;
    const int n_lane = owns ? want[i] : 0;
    epa::Arena da = as.a[dst];
    const long long t = da.warp_try_take(n_lane);
    const int taken = t < 0 ? 0 : n_lane;
    int incl = taken;
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
    }
    const int tot = __shfl_sync(0xffffffffu, incl, 31), excl = incl - taken;
    for (int q0 = 0; q0 < tot; q0 += 32) {
        const int q = q0 + lane;
        int own = -1;
        for (int l = 0; l < 32; ++l) {
            const int e = __shfl_sync(0xffffffffu, excl, l), n = __shfl_sync(0xffffffffu, taken, l);
            if (q < tot and e <= q and q < e + n) own = l;
        }
        const int so = own < 0 ? 0 : own;
        const long long ts = __shfl_sync(0xffffffffu, t, so);
        const int es = __shfl_sync(0xffffffffu, excl, so), ds = __shfl_sync(0xffffffffu, dst, so);
        const int els = __shfl_sync(0xffffffffu, el, so), is = __shfl_sync(0xffffffffu, i, so);
        if (own >= 0) {
            const int id = as.a[ds].page_at(static_cast<unsigned long long>(ts), q - es);
            ltable[is * kMaxPages + q - es] = id;
            rtab_base[ds][(els * R + src) * kMaxPages + q - es] = id;
        }
        __syncwarp();
    }
    if (owns) got[i] = taken;
    __threadfence_system();
    __syncwarp();
    if (lane == 0) span[blockIdx.x * 2] = s, span[blockIdx.x * 2 + 1] = gtime();
}
// receiver: 48 blocks x 8 warps, warp = one (local expert, source) key
__global__ void mc_give(epa::Arena a, const int* rtable, const int* taken, unsigned long long* span, const volatile int* go) {
    if (threadIdx.x == 0) wait_go(go);
    __syncthreads();
    const unsigned long long s = gtime();
    const int key = blockIdx.x * 8 + threadIdx.x / 32;   // el * R + src
    if (taken[key] > 0) warp_give_full(a, rtable + key * kMaxPages, taken[key]);
    __syncthreads();
    if (threadIdx.x == 0) span[blockIdx.x * 2] = s, span[blockIdx.x * 2 + 1] = gtime();
}

// ------------------------------------------------------------------ distmoe
__global__ void dm_take(epa::Arena a, int* page, long long* cyc) {
    const long long t0 = clock64();
    *page = a.page_at(a.wait_take(1), 0);
    cyc[0] = clock64() - t0;
}
__global__ void dm_give(epa::Arena a, const int* page, long long* cyc) {
    const long long t0 = clock64();
    if (*page >= 0) a.give(*page);
    cyc[1] = clock64() - t0;
}
__global__ void dm_empty(int* page, long long* cyc) { if (page == nullptr) cyc[0] = 0; }

// ------------------------------------------------------------------ mega (modeled)
__global__ void mega_kernel(epa::Arena a, int blocks, long long* lat, unsigned long long* span, const volatile int* go, int warm) {
    if (threadIdx.x != 0) return;
    wait_go(go);
    const unsigned long long s = gtime();   // the wall includes any warm-up below
    if (warm == 1) __threadfence_system();                                   // pay the first system fence before timing
    if (warm == 2) atomicAdd_system(reinterpret_cast<int*>(&a.meta()->failed) + 1, 0);   // touch the ring state (adds 0)
    if (warm == 3) { __threadfence_system(); atomicAdd_system(reinterpret_cast<int*>(&a.meta()->failed) + 1, 0); __threadfence_system(); }
    for (int b = blockIdx.x; b < blocks; b += gridDim.x) {
        const long long t0 = clock64();
        const int p = a.page_at(a.wait_take(1), 0);
        const long long t1 = clock64();
        a.give(p);
        const long long t2 = clock64();
        lat[b * 2] = t1 - t0, lat[b * 2 + 1] = t2 - t1;
    }
    span[blockIdx.x * 2] = s, span[blockIdx.x * 2 + 1] = gtime();
}

// ------------------------------------------------------------------ fence costs
// one thread, 256 times each: out[k] = cycles per op
__global__ void fence_kernel(unsigned long long* x, long long* out) {
    using S = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
    using G = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>;
    long long t;
    unsigned long long acc = 0;
#define TIME(k, stmt) t = clock64(); for (int i = 0; i < 256; ++i) { stmt; } out[k] = (clock64() - t) / 256;
    TIME(0, S(x[i & 7]).store(i, cuda::memory_order_relaxed))
    TIME(1, G(x[i & 7]).store(i, cuda::memory_order_release))
    TIME(2, S(x[i & 7]).store(i, cuda::memory_order_release))
    TIME(3, acc += G(x[i & 7]).load(cuda::memory_order_acquire))
    TIME(4, acc += S(x[i & 7]).load(cuda::memory_order_acquire))
    TIME(5, acc += S(x[i & 7]).load(cuda::memory_order_relaxed))
    TIME(6, acc += S(x[0]).fetch_add(1, cuda::memory_order_relaxed))
    TIME(7, acc += S(x[0]).fetch_add(1, cuda::memory_order_release))
    TIME(8, __threadfence())
    TIME(9, __threadfence_system())
#undef TIME
    out[10] = static_cast<long long>(acc & 1);
}

// ================================================================== host
static int ndev;
static epa::Arena arena[R];
static int* go;

static void init_arena(int d, int pages) {
    CHECK(cudaSetDevice(d));
    arena[d].num_pages = pages;
    epa::arena_init_kernel<<<64, 256>>>(arena[d]);
    CHECK(cudaDeviceSynchronize());
}
static void sync_all() { for (int d = 0; d < ndev; ++d) { CHECK(cudaSetDevice(d)); CHECK(cudaDeviceSynchronize()); } }
static void release() { __atomic_store_n(go, 1, __ATOMIC_SEQ_CST); }
static void arm() { *go = getenv("NOGO") ? 1 : 0; }

template <typename T> static std::vector<T> d2h(int d, const T* p, size_t n) {
    std::vector<T> v(n);
    CHECK(cudaSetDevice(d));
    CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static double q(std::vector<double> v, double f) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[std::min<size_t>(v.size() - 1, static_cast<size_t>(f * v.size()))];
}
static double ghz_of(const long long* c) { return double(c[2] - c[0]) / double(c[3] - c[1]); }

static void run_ops(const char* where, int dev, int arena_dev, int n) {
    const int iters = 400;
    init_arena(arena_dev, 4096);
    CHECK(cudaSetDevice(dev));
    long long *out, *clk;
    CHECK(cudaMalloc(&out, iters * 6 * 8));
    CHECK(cudaMalloc(&clk, 4 * 8));
    ops_kernel<<<1, 32>>>(arena[arena_dev], n, iters, out, clk);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    auto o = d2h(dev, out, iters * 6);
    auto c = d2h(dev, clk, 4);
    const double g = ghz_of(c.data());
    std::vector<double> col[6];
    for (int it = 20; it < iters; ++it)
        for (int k = 0; k < 6; ++k) if (o[it * 6 + k] >= 0) col[k].push_back(o[it * 6 + k] / g);
    printf("{\"test\":\"ops\",\"where\":\"%s\",\"n\":%d,\"ghz\":%.3f,\"reserve\":%.0f,\"take_reserved\":%.0f,"
           "\"page_at_1thread\":%.0f,\"give_1thread\":%.0f,\"page_at_warp\":%.0f,\"warp_give\":%.0f,"
           "\"reserve_p99\":%.0f,\"give_1thread_p99\":%.0f}\n",
           where, n, g, q(col[0], .5), q(col[1], .5), q(col[2], .5), q(col[3], .5), q(col[4], .5), q(col[5], .5),
           q(col[0], .99), q(col[3], .99));
    CHECK(cudaFree(out)); CHECK(cudaFree(clk));
}

// warps on each GPU in `devs` loop on arena_dev's arena
static void run_conc(const char* where, std::vector<int> devs, int arena_dev, int S, int W) {
    const int iters = S * W >= 512 ? 64 : 256;
    init_arena(arena_dev, 32768);
    const long long warps = static_cast<long long>(S) * W;
    std::vector<long long*> lat(devs.size()), clk(devs.size());
    std::vector<unsigned long long*> span(devs.size());
    arm();
    for (size_t i = 0; i < devs.size(); ++i) {
        CHECK(cudaSetDevice(devs[i]));
        CHECK(cudaMalloc(&lat[i], warps * iters * 2 * 8));
        CHECK(cudaMalloc(&span[i], warps * 2 * 8));
        CHECK(cudaMalloc(&clk[i], 4 * 8));
        conc_kernel<<<S, 32 * W>>>(arena[arena_dev], iters, lat[i], span[i], go, clk[i]);
        CHECK(cudaGetLastError());
    }
    release();
    sync_all();
    std::vector<double> take, give;
    double secs = 0;
    long long ok = 0, failed = 0;
    for (size_t i = 0; i < devs.size(); ++i) {
        unsigned long long lo = ~0ull, hi = 0;   // globaltimer is per GPU: compare only within one
        auto l = d2h(devs[i], lat[i], warps * iters * 2);
        auto s = d2h(devs[i], span[i], warps * 2);
        auto c = d2h(devs[i], clk[i], 4);
        const double g = ghz_of(c.data());
        for (long long w = 0; w < warps; ++w) {
            lo = std::min(lo, s[w * 2]), hi = std::max(hi, s[w * 2 + 1]);
            for (int k = 0; k < iters; ++k) {
                const long long a = l[(w * iters + k) * 2], b = l[(w * iters + k) * 2 + 1];
                if (a < 0) { ++failed; continue; }
                ++ok;
                if (k >= 4) take.push_back(a / g), give.push_back(b / g);
            }
        }
        secs = std::max(secs, (hi - lo) * 1e-9);
        CHECK(cudaSetDevice(devs[i]));
        CHECK(cudaFree(lat[i])); CHECK(cudaFree(span[i])); CHECK(cudaFree(clk[i]));
    }
    printf("{\"test\":\"conc\",\"where\":\"%s\",\"gpus\":%zu,\"sms\":%d,\"warps_per_sm\":%d,\"take_med\":%.0f,\"take_p99\":%.0f,"
           "\"give_med\":%.0f,\"give_p99\":%.0f,\"pairs_per_us\":%.2f,\"failed\":%lld}\n",
           where, devs.size(), S, W, q(take, .5), q(take, .99), q(give, .5), q(give, .99), ok / secs * 1e-6, failed);
}

static std::vector<std::vector<int>> load(const std::string& f, int cols) {
    std::ifstream in(f);
    std::vector<std::vector<int>> m(R, std::vector<int>(cols));
    for (auto& r : m) for (auto& x : r) in >> x;
    if (!in) { printf("cannot read %s\n", f.c_str()); exit(1); }
    return m;
}

static void run_deepep(const std::string& dir, const char* routing, int pages) {
    auto want = load(dir + "/deepep_" + routing + ".txt", R);
    for (int d = 0; d < R; ++d) init_arena(d, pages);
    int *dwant[R], *dgot[R], *ltab[R], *rtab[R], **rtp[R], *taken[R];
    unsigned long long *tt[R], *tg[R];
    Arenas as;
    for (int d = 0; d < R; ++d) as.a[d] = arena[d];
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaMalloc(&dwant[d], R * 4)); CHECK(cudaMalloc(&dgot[d], R * 4)); CHECK(cudaMalloc(&taken[d], R * 4));
        CHECK(cudaMalloc(&ltab[d], R * kMaxPages * 4)); CHECK(cudaMalloc(&rtab[d], R * kMaxPages * 4));
        CHECK(cudaMalloc(&rtp[d], R * 8)); CHECK(cudaMalloc(&tt[d], 16)); CHECK(cudaMalloc(&tg[d], 16));
        CHECK(cudaMemcpy(dwant[d], want[d].data(), R * 4, cudaMemcpyHostToDevice));
    }
    for (int s = 0; s < R; ++s) {
        std::vector<int*> p(R);
        for (int d = 0; d < R; ++d) p[d] = rtab[d] + s * kMaxPages;   // row s of destination d's rtable
        CHECK(cudaSetDevice(s));
        CHECK(cudaMemcpy(rtp[s], p.data(), R * 8, cudaMemcpyHostToDevice));
    }
    std::vector<double> take_atomics, take_all, give;
    long long got_pages = 0, want_pages = 0;
    for (int call = 0; call < 60; ++call) {
        arm();
        for (int s = 0; s < R; ++s) { CHECK(cudaSetDevice(s)); deepep_take<<<1, 256>>>(as, dwant[s], dgot[s], ltab[s], rtp[s], tt[s], go); }
        release();
        sync_all();
        std::vector<std::vector<int>> got(R);
        double ma = 0, mt = 0;
        for (int s = 0; s < R; ++s) {
            got[s] = d2h(s, dgot[s], R);
            auto t = d2h(s, tt[s], 2);
            ma = std::max(ma, t[0] / 1e3), mt = std::max(mt, t[1] / 1e3);
        }
        for (int d = 0; d < R; ++d) {
            std::vector<int> col(R);
            for (int s = 0; s < R; ++s) col[s] = got[s][d], got_pages += got[s][d], want_pages += want[s][d];
            CHECK(cudaSetDevice(d));
            CHECK(cudaMemcpy(taken[d], col.data(), R * 4, cudaMemcpyHostToDevice));
        }
        arm();
        for (int d = 0; d < R; ++d) { CHECK(cudaSetDevice(d)); deepep_give<<<1, 256>>>(arena[d], rtab[d], taken[d], tg[d], go); }
        release();
        sync_all();
        double mg = 0;
        for (int d = 0; d < R; ++d) mg = std::max(mg, d2h(d, tg[d], 1)[0] / 1e3);
        if (call >= 10) take_atomics.push_back(ma), take_all.push_back(mt), give.push_back(mg);
    }
    printf("{\"test\":\"deepep\",\"routing\":\"%s\",\"pages\":%d,\"take_atomics_us\":%.2f,\"take_us\":%.2f,\"take_us_p90\":%.2f,"
           "\"give_us\":%.2f,\"give_us_p90\":%.2f,\"got_frac\":%.4f}\n",
           routing, pages, q(take_atomics, .5), q(take_all, .5), q(take_all, .9), q(give, .5), q(give, .9),
           double(got_pages) / want_pages);
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        cudaFree(dwant[d]); cudaFree(dgot[d]); cudaFree(taken[d]); cudaFree(ltab[d]); cudaFree(rtab[d]);
        cudaFree(rtp[d]); cudaFree(tt[d]); cudaFree(tg[d]);
    }
}

static void run_mooncake(const std::string& dir, const char* routing, int pages) {
    auto want = load(dir + "/mc_" + routing + ".txt", E);
    for (int d = 0; d < R; ++d) init_arena(d, pages);
    Arenas as;
    for (int d = 0; d < R; ++d) as.a[d] = arena[d];
    int *dwant[R], *dgot[R], *ltab[R], *rtab[R], **rtp[R], *taken[R];
    unsigned long long *st[R], *sg[R];
    const int nk = (E / R) * R;   // receiver keys
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaMalloc(&dwant[d], E * 4)); CHECK(cudaMalloc(&dgot[d], E * 4)); CHECK(cudaMalloc(&taken[d], nk * 4));
        CHECK(cudaMalloc(&ltab[d], size_t(E) * kMaxPages * 4)); CHECK(cudaMalloc(&rtab[d], size_t(nk) * kMaxPages * 4));
        CHECK(cudaMalloc(&rtp[d], R * 8)); CHECK(cudaMalloc(&st[d], 48 * 16)); CHECK(cudaMalloc(&sg[d], 48 * 16));
        CHECK(cudaMemcpy(dwant[d], want[d].data(), E * 4, cudaMemcpyHostToDevice));
    }
    for (int s = 0; s < R; ++s) {
        CHECK(cudaSetDevice(s));
        CHECK(cudaMemcpy(rtp[s], rtab, R * 8, cudaMemcpyHostToDevice));
    }
    auto wall = [](const std::vector<unsigned long long>& v, int n) {
        unsigned long long lo = ~0ull, hi = 0;
        for (int i = 0; i < n; ++i) lo = std::min(lo, v[i * 2]), hi = std::max(hi, v[i * 2 + 1]);
        return (hi - lo) / 1e3;
    };
    std::vector<double> take, give;
    long long got_pages = 0, want_pages = 0;
    for (int call = 0; call < 60; ++call) {
        arm();
        for (int s = 0; s < R; ++s) { CHECK(cudaSetDevice(s)); mc_take<<<48, 32>>>(as, s, dwant[s], dgot[s], ltab[s], rtp[s], st[s], go); }
        release();
        sync_all();
        double mt = 0;
        std::vector<std::vector<int>> got(R);
        for (int s = 0; s < R; ++s) got[s] = d2h(s, dgot[s], E), mt = std::max(mt, wall(d2h(s, st[s], 96), 48));
        for (int d = 0; d < R; ++d) {
            std::vector<int> t(nk);
            for (int el = 0; el < E / R; ++el)
                for (int s = 0; s < R; ++s) {
                    const int e = d * (E / R) + el;
                    t[el * R + s] = got[s][e], got_pages += got[s][e], want_pages += want[s][e];
                }
            CHECK(cudaSetDevice(d));
            CHECK(cudaMemcpy(taken[d], t.data(), nk * 4, cudaMemcpyHostToDevice));
        }
        arm();
        for (int d = 0; d < R; ++d) { CHECK(cudaSetDevice(d)); mc_give<<<nk / 8, 256>>>(arena[d], rtab[d], taken[d], sg[d], go); }
        release();
        sync_all();
        double mg = 0;
        for (int d = 0; d < R; ++d) mg = std::max(mg, wall(d2h(d, sg[d], 96), 48));
        if (call >= 10) take.push_back(mt), give.push_back(mg);
    }
    printf("{\"test\":\"mooncake\",\"routing\":\"%s\",\"pages\":%d,\"take_us\":%.2f,\"take_us_p90\":%.2f,\"give_us\":%.2f,"
           "\"give_us_p90\":%.2f,\"got_frac\":%.4f}\n",
           routing, pages, q(take, .5), q(take, .9), q(give, .5), q(give, .9), double(got_pages) / want_pages);
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        cudaFree(dwant[d]); cudaFree(dgot[d]); cudaFree(taken[d]); cudaFree(ltab[d]); cudaFree(rtab[d]);
        cudaFree(rtp[d]); cudaFree(st[d]); cudaFree(sg[d]);
    }
}

static void run_distmoe() {
    init_arena(0, 4);
    CHECK(cudaSetDevice(0));
    int* page;
    long long* cyc;
    CHECK(cudaMalloc(&page, 4));
    CHECK(cudaMalloc(&cyc, 16));
    cudaStream_t st;
    CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const int rounds = 100;
    auto graph_us = [&](bool empty) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int r = 0; r < rounds; ++r) {
            if (empty) { dm_empty<<<1, 1, 0, st>>>(page, cyc); dm_empty<<<1, 1, 0, st>>>(page, cyc); }
            else { dm_take<<<1, 1, 0, st>>>(arena[0], page, cyc); dm_give<<<1, 1, 0, st>>>(arena[0], page, cyc); }
        }
        CHECK(cudaStreamEndCapture(st, &g));
        CHECK(cudaGraphInstantiate(&ge, g, 0));
        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        std::vector<double> v;
        for (int rep = 0; rep < 30; ++rep) {
            CHECK(cudaEventRecord(a, st));
            CHECK(cudaGraphLaunch(ge, st));
            CHECK(cudaEventRecord(b, st));
            CHECK(cudaEventSynchronize(b));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, a, b));
            if (rep >= 5) v.push_back(ms * 1e3 / rounds);
        }
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        return q(v, .5);
    };
    const double real = graph_us(false), empty = graph_us(true);
    // in-kernel cycles of one take and one give
    std::vector<double> t, g;
    long long *clkd, *scratch;
    CHECK(cudaMalloc(&clkd, 32));
    CHECK(cudaMalloc(&scratch, 4 * 6 * 8));
    ops_kernel<<<1, 32>>>(arena[0], 1, 4, scratch, clkd);   // only for the clock rate
    CHECK(cudaDeviceSynchronize());
    const double ghz = ghz_of(d2h(0, clkd, 4).data());
    for (int r = 0; r < 200; ++r) {
        dm_take<<<1, 1>>>(arena[0], page, cyc);
        dm_give<<<1, 1>>>(arena[0], page, cyc);
        auto c = d2h(0, cyc, 2);
        if (r >= 10) t.push_back(c[0] / ghz), g.push_back(c[1] / ghz);
    }
    printf("{\"test\":\"distmoe\",\"round_us\":%.2f,\"empty_round_us\":%.2f,\"take_ns\":%.0f,\"give_ns\":%.0f}\n", real, empty,
           q(t, .5), q(g, .5));
}

static void run_mega(int blocks, int pages) {
    for (int d = 0; d < R; ++d) init_arena(d, pages);
    long long* lat[R];
    unsigned long long* span[R];
    const int sms = 148;
    std::vector<double> wall, take, give;
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaMalloc(&lat[d], blocks * 16)); CHECK(cudaMalloc(&span[d], sms * 16));
    }
    long long clk_h[4];
    {
        long long *o, *c;
        CHECK(cudaSetDevice(0));
        CHECK(cudaMalloc(&o, 4 * 6 * 8)); CHECK(cudaMalloc(&c, 32));
        ops_kernel<<<1, 32>>>(arena[0], 1, 4, o, c);
        CHECK(cudaDeviceSynchronize());
        auto v = d2h(0, c, 4);
        std::copy(v.begin(), v.end(), clk_h);
        CHECK(cudaFree(o)); CHECK(cudaFree(c));
    }
    const double g = ghz_of(clk_h);
    for (int call = 0; call < 40; ++call) {
        arm();
        for (int d = 0; d < R; ++d) { CHECK(cudaSetDevice(d)); mega_kernel<<<sms, 32>>>(arena[d], blocks, lat[d], span[d], go, getenv("MEGA_WARM") ? atoi(getenv("MEGA_WARM")) : 0); }
        release();
        sync_all();
        double mw = 0;
        for (int d = 0; d < R; ++d) {
            auto s = d2h(d, span[d], sms * 2);
            unsigned long long lo = ~0ull, hi = 0;
            for (int i = 0; i < sms; ++i) lo = std::min(lo, s[i * 2]), hi = std::max(hi, s[i * 2 + 1]);
            mw = std::max(mw, (hi - lo) / 1e3);
            if (call >= 5) {
                auto l = d2h(d, lat[d], blocks * 2);
                for (int b = 0; b < blocks; ++b) take.push_back(l[b * 2] / g), give.push_back(l[b * 2 + 1] / g);
            }
        }
        if (call >= 5) wall.push_back(mw);
    }
    printf("{\"test\":\"mega\",\"blocks\":%d,\"pages\":%d,\"wall_us\":%.2f,\"wall_us_p90\":%.2f,\"take_med\":%.0f,\"take_p99\":%.0f,"
           "\"give_med\":%.0f,\"give_p99\":%.0f}\n",
           blocks, pages, q(wall, .5), q(wall, .9), q(take, .5), q(take, .99), q(give, .5), q(give, .99));
    for (int d = 0; d < R; ++d) { CHECK(cudaSetDevice(d)); cudaFree(lat[d]); cudaFree(span[d]); }
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    setvbuf(stdout, nullptr, _IOLBF, 0);
    CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < R) { printf("needs %d GPUs\n", R); return 1; }
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        for (int p = 0; p < R; ++p) if (p != d) cudaDeviceEnablePeerAccess(p, 0);
        cudaGetLastError();
        arena[d].page_bytes = 4096;   // take/give never touch page data
        CHECK(cudaMalloc(&arena[d].base, epa::Arena::bytes_for(32768, arena[d].page_bytes)));
    }
    CHECK(cudaHostAlloc(&go, 4, cudaHostAllocMapped | cudaHostAllocPortable));

    const std::string only = getenv("ONLY") ? getenv("ONLY") : "ops,conc,hot,deepep,mooncake,distmoe,mega";
    auto want = [&](const char* t) { return only.find(t) != std::string::npos; };
    if (want("ops")) for (int n : (getenv("OPS_N") ? std::vector<int>{atoi(getenv("OPS_N"))} : std::vector<int>{1, 8, 36, 64})) {
        fprintf(stderr, "ops n=%d\n", n);
        run_ops("same GPU", 0, 0, n);
        run_ops("other GPU", 0, 1, n);
    }
    if (want("conc")) for (int S : {1, 16, 64, 148})
        for (int W : {1, 4, 16}) {
            run_conc("same GPU", {0}, 0, S, W);
            run_conc("other GPU", {0}, 1, S, W);
        }
    if (want("hot")) for (int S : {1, 16, 148}) run_conc("8 GPUs on GPU 0's arena", {0, 1, 2, 3, 4, 5, 6, 7}, 0, S, 4);
    for (const char* r : {"uniform", "extreme"}) {
        if (want("deepep")) run_deepep(dir, r, 363), run_deepep(dir, r, 576);
        if (want("mooncake")) run_mooncake(dir, r, 3456);
    }
    if (want("fence")) {
        for (int d : {0, 1}) {
            CHECK(cudaSetDevice(0));
            unsigned long long* x;
            long long *o, *c, *scratch;
            CHECK(cudaSetDevice(d)); CHECK(cudaMalloc(&x, 64)); CHECK(cudaMemset(x, 0, 64));
            CHECK(cudaSetDevice(0)); CHECK(cudaMalloc(&o, 11 * 8)); CHECK(cudaMalloc(&c, 32)); CHECK(cudaMalloc(&scratch, 4 * 6 * 8));
            ops_kernel<<<1, 32>>>(arena[0], 1, 4, scratch, c);
            fence_kernel<<<1, 1>>>(x, o);
            CHECK(cudaDeviceSynchronize());
            auto v = d2h(0, o, 11);
            const double g = ghz_of(d2h(0, c, 4).data());
            printf("{\"test\":\"fence\",\"where\":\"%s\",\"st_relaxed_sys\":%.0f,\"st_release_gpu\":%.0f,\"st_release_sys\":%.0f,"
                   "\"ld_acquire_gpu\":%.0f,\"ld_acquire_sys\":%.0f,\"ld_relaxed_sys\":%.0f,\"fadd_relaxed_sys\":%.0f,"
                   "\"fadd_release_sys\":%.0f,\"threadfence\":%.0f,\"threadfence_system\":%.0f}\n",
                   d == 0 ? "same GPU" : "other GPU", v[0] / g, v[1] / g, v[2] / g, v[3] / g, v[4] / g, v[5] / g, v[6] / g, v[7] / g,
                   v[8] / g, v[9] / g);
            CHECK(cudaFree(o)); CHECK(cudaFree(c)); CHECK(cudaFree(scratch));
            CHECK(cudaSetDevice(d)); CHECK(cudaFree(x));
        }
    }
    if (want("distmoe")) run_distmoe();
    if (want("mega")) {
        run_mega(142, 363);
        if (getenv("MEGA_MORE")) run_mega(148 * 16, 363);
    }
    return 0;
}
