// Latency of each epa::WarpHeap operation, timed with clock64() from the calling warp.
// Build: nvcc -std=c++17 -O3 -arch=sm_100a -I ../include bench_heap_latency.cu -o bench_heap_latency
//
// Operations (one sample = one call as the caller sees it):
//   alloc, page hit     all 32 lanes call alloc(); the warp bumps inside the page it owns
//   alloc, page miss    the owned page is full: retire it, take a new page from the ring
//   retire              drop the owner reference on the current page
//   free, one thread    one lane frees one object (one system-scope atomic)
//   free, last object   that free drops the page's count to 0 and gives the page back
//   free, whole warp    32 lanes free at once (objects mostly on the same page)
//   ring take / give    try_take(1) + page_at, and give(1), for reference
// Conditions: one warp alone; 148 x 8 warps doing the same at once; one warp on GPU 1
// using GPU 0's arena over NVLink.
#include <epa/arena.cuh>

#include <algorithm>
#include <cstdio>
#include <vector>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

enum Op { kAllocHit, kAllocMiss, kRetire, kFreeOne, kFreeLast, kFreeWarp, kTake, kGive, kNumOps };
static const char* kOpName[kNumOps] = {"alloc, page hit", "alloc, page miss (takes a page)", "retire",
                                       "free, one thread", "free, last object (gives the page)",
                                       "free, whole warp at once", "ring try_take(1) + page_at", "ring give(1)"};

// Per-warp sample slots (no shared counter, so recording does not perturb other warps).
struct Samples {
    long long* cyc;
    int* op;
    int per_warp;
    long long* clk;   // warp 0: clock64 and globaltimer at start and end (SM clock rate)
};

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void clk_mark(const Samples& s, long long warp, int at) {
    if (warp == 0 and (threadIdx.x & 31) == 0) s.clk[2 * at] = clock64(), s.clk[2 * at + 1] = static_cast<long long>(gtime());
}

__global__ void alloc_kernel(epa::Arena a, int iters, int bytes, void** ptrs, Samples s) {
    const int lane = threadIdx.x & 31;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    long long* cyc = s.cyc + warp * s.per_warp;
    int* op = s.op + warp * s.per_warp;
    int k = 0;
    clk_mark(s, warp, 0);
    epa::WarpHeap h{a};
    for (int i = 0; i < iters; ++i) {
        const int before = h.page;
        __syncwarp();
        const long long t0 = clock64();
        void* p = h.alloc(bytes);
        __syncwarp();
        const long long t1 = clock64();
        ptrs[(warp * iters + i) * 32 + lane] = p;
        if (lane == 0) cyc[k] = t1 - t0, op[k] = (before < 0 or h.page != before) ? kAllocMiss : kAllocHit;
        ++k;
    }
    __syncwarp();
    const long long t0 = clock64();
    h.retire();
    __syncwarp();
    const long long t1 = clock64();
    if (lane == 0) cyc[k] = t1 - t0, op[k] = kRetire;
    clk_mark(s, warp, 1);
}

// Odd rows: lanes free one at a time (single-thread latency). Even rows: all lanes at once.
// A page's last row is odd for both object sizes, so its last free is timed alone.
// Each warp frees only its own objects after retiring, so refs[page] == 1 before a free
// means exactly that this free is the page's last.
__global__ void free_kernel(epa::Arena a, int iters, void** ptrs, Samples s) {
    const int lane = threadIdx.x & 31;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    long long* cyc = s.cyc + warp * s.per_warp;
    int* op = s.op + warp * s.per_warp;
    int k = 0;
    clk_mark(s, warp, 0);
    for (int i = 0; i < iters; ++i) {
        void** row = ptrs + (warp * iters + i) * 32;
        if (i % 2 == 1) {
            for (int l = 0; l < 32; ++l) {
                if (lane == l) {
                    void* p = row[l];
                    const bool last = *reinterpret_cast<volatile int*>(&a.refs()[a.page_of(p)]) == 1;
                    const long long t0 = clock64();
                    epa::WarpHeap::free(a, p);
                    const long long t1 = clock64();
                    cyc[k] = t1 - t0, op[k] = last ? kFreeLast : kFreeOne;
                }
                ++k;
                __syncwarp();
            }
        } else {
            void* p = row[lane];
            __syncwarp();
            const long long t0 = clock64();
            epa::WarpHeap::free(a, p);
            __syncwarp();
            const long long t1 = clock64();
            if (lane == 0) cyc[k] = t1 - t0, op[k] = kFreeWarp;
            ++k;
        }
    }
    clk_mark(s, warp, 1);
}

__global__ void ring_kernel(epa::Arena a, int iters, Samples s) {
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if ((threadIdx.x & 31) != 0) return;
    long long* cyc = s.cyc + warp * s.per_warp;
    int* op = s.op + warp * s.per_warp;
    clk_mark(s, warp, 0);
    for (int i = 0; i < iters; ++i) {
        long long t0 = clock64();
        const long long t = a.try_take(1);
        const int p = t < 0 ? -1 : a.page_at(static_cast<unsigned long long>(t), 0);
        long long t1 = clock64();
        cyc[2 * i] = t1 - t0, op[2 * i] = t < 0 ? -1 : kTake;
        t0 = clock64();
        if (p >= 0) a.give(p);
        t1 = clock64();
        cyc[2 * i + 1] = t1 - t0, op[2 * i + 1] = p < 0 ? -1 : kGive;
    }
    clk_mark(s, warp, 1);
}

struct Result {
    std::vector<double> med, p99;
    std::vector<long long> n;
};

static void collect(int dev, const Samples& s, long long slots, double ghz, std::vector<std::vector<double>>& ns) {
    std::vector<long long> cyc(slots);
    std::vector<int> op(slots);
    CHECK(cudaSetDevice(dev));
    CHECK(cudaMemcpy(cyc.data(), s.cyc, slots * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(op.data(), s.op, slots * 4, cudaMemcpyDeviceToHost));
    for (long long i = 0; i < slots; ++i)
        if (op[i] >= 0 and op[i] < kNumOps) ns[op[i]].push_back(cyc[i] / ghz);
}

static double clock_ghz(int dev, const Samples& s) {
    long long c[4];
    CHECK(cudaSetDevice(dev));
    CHECK(cudaMemcpy(c, s.clk, sizeof c, cudaMemcpyDeviceToHost));
    return double(c[2] - c[0]) / double(c[3] - c[1]);
}

// One condition: warps on device `dev` use the arena `a` (on device 0).
static void run(const char* name, int dev, const epa::Arena& a, int blocks, int threads, int iters, int bytes) {
    const long long warps = static_cast<long long>(blocks) * threads / 32;
    CHECK(cudaSetDevice(dev));
    void** ptrs;
    Samples s;
    s.per_warp = std::max(iters + 1, iters / 2 * 33 + iters);
    const long long slots = warps * s.per_warp;
    CHECK(cudaMalloc(&ptrs, warps * iters * 32 * 8));
    CHECK(cudaMalloc(&s.cyc, slots * 8));
    CHECK(cudaMalloc(&s.op, slots * 4));
    CHECK(cudaMalloc(&s.clk, 4 * 8));

    std::vector<std::vector<double>> ns(kNumOps);
    double ghz = 0;
    for (int rep = 0; rep < 3; ++rep) {   // rep 0 warms up
        for (int phase = 0; phase < 3; ++phase) {
            CHECK(cudaMemset(s.op, 0xff, slots * 4));
            if (phase == 0) alloc_kernel<<<blocks, threads>>>(a, iters, bytes, ptrs, s);
            if (phase == 1) free_kernel<<<blocks, threads>>>(a, iters, ptrs, s);
            if (phase == 2) ring_kernel<<<blocks, threads>>>(a, iters, s);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            const double g = clock_ghz(dev, s);
            if (rep > 0) collect(dev, s, slots, g, ns), ghz = g;
        }
    }
    // every page must be back
    epa::ArenaMeta m;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMemcpy(&m, a.base, sizeof m, cudaMemcpyDeviceToHost));

    printf("\n%s  (%lld warp%s, %d B objects, SM clock %.2f GHz, pages back: %s, failed takes: %llu)\n", name, warps,
           warps > 1 ? "s" : "", bytes, ghz, m.free_pages == a.num_pages ? "all" : "NO", m.failed);
    printf("  %-38s %10s %10s %10s %10s\n", "operation", "median ns", "p10 ns", "p99 ns", "samples");
    for (int o = 0; o < kNumOps; ++o) {
        auto& v = ns[o];
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        auto q = [&](double f) { return v[std::min<size_t>(v.size() - 1, static_cast<size_t>(f * v.size()))]; };
        printf("  %-38s %10.0f %10.0f %10.0f %10zu\n", kOpName[o], q(0.5), q(0.1), q(0.99), v.size());
    }
    CHECK(cudaSetDevice(dev));
    CHECK(cudaFree(ptrs)); CHECK(cudaFree(s.cyc)); CHECK(cudaFree(s.op)); CHECK(cudaFree(s.clk));
}

int main() {
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    CHECK(cudaSetDevice(0));
    epa::Arena a;
    a.num_pages = 65536, a.page_bytes = 64 << 10;   // 4 GiB of 64 KiB pages
    CHECK(cudaMalloc(&a.base, epa::Arena::bytes_for(a.num_pages, a.page_bytes)));
    epa::arena_init_kernel<<<64, 256>>>(a);
    CHECK(cudaDeviceSynchronize());
    printf("epa::WarpHeap latency, %d pages of %lld KiB\n", a.num_pages, static_cast<long long>(a.page_bytes >> 10));

    run("GPU 0, one warp alone", 0, a, 1, 32, 4096, 16);
    run("GPU 0, one warp alone", 0, a, 1, 32, 1024, 1024);
    run("GPU 0, 148 x 8 warps at once", 0, a, 148, 256, 512, 16);
    run("GPU 0, 148 x 8 warps at once", 0, a, 148, 256, 64, 1024);
    if (ndev >= 2) {
        int can = 0;
        CHECK(cudaDeviceCanAccessPeer(&can, 1, 0));
        if (can) {
            CHECK(cudaSetDevice(1));
            cudaDeviceEnablePeerAccess(0, 0);
            CHECK(cudaSetDevice(0));
            cudaDeviceEnablePeerAccess(1, 0);
            cudaGetLastError();
            run("GPU 1 using GPU 0's arena over NVLink, one warp", 1, a, 1, 32, 4096, 16);
            run("GPU 1 using GPU 0's arena over NVLink, one warp", 1, a, 1, 32, 1024, 1024);
        }
    }
    return 0;
}
