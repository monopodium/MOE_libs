// epa::Arena as a general in-kernel allocator (no EP involved).
//   1. malloc/free throughput: WarpHeap vs CUDA's device malloc
//   2. page take/give throughput, local and from a peer GPU over NVLink
//   3. a kernel with data-dependent output size: worst-case preallocation vs
//      count + scan + write vs one pass with PagedArray::append
// Build: nvcc -std=c++17 -O3 -arch=sm_100a -I ../include bench_arena.cu -o bench_arena
#include <algorithm>
#include <cstdio>
#include <vector>

#include <cub/cub.cuh>
#include <epa/arena.cuh>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ unsigned hash(unsigned x) {
    x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
    return x;
}

epa::Arena make_arena(int device, int num_pages, int64_t page_bytes) {
    CHECK(cudaSetDevice(device));
    char* mem;
    CHECK(cudaMalloc(&mem, epa::Arena::bytes_for(num_pages, page_bytes)));
    epa::Arena a;
    a.base = mem, a.page_bytes = page_bytes, a.num_pages = num_pages;
    epa::arena_init_kernel<<<64, 256>>>(a);
    CHECK(cudaDeviceSynchronize());
    return a;
}

template <typename F>
float time_ms(F f, int reps = 5) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    f();
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaEventRecord(a));
        f();
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

// ------------------------------------------------------------------ 1. malloc/free
// Each thread: `rounds` x (allocate `size` bytes, write 16 B, keep it for `keep`
// rounds, free). keep > 0 keeps several objects live per thread.
template <bool kEPA>
__global__ void malloc_kernel(epa::Arena a, int size, int rounds, int* sink) {
    constexpr int keep = 4;
    void* live[keep] = {};
    epa::WarpHeap h{a};
    for (int r = 0; r < rounds; ++r) {
        void*& slot = live[r % keep];
        if (slot != nullptr) {
            if constexpr (kEPA) epa::WarpHeap::free(a, slot);
            else free(slot);
        }
        void* p;
        if constexpr (kEPA) p = h.alloc(size, true);
        else p = malloc(size);
        if (p == nullptr) { atomicAdd(sink, 1); slot = nullptr; continue; }
        *static_cast<int4*>(p) = make_int4(r, r, r, r);
        slot = p;
    }
    for (int i = 0; i < keep; ++i)
        if (live[i] != nullptr) {
            if constexpr (kEPA) epa::WarpHeap::free(a, live[i]);
            else free(live[i]);
        }
    if constexpr (kEPA) h.retire();
}

// ------------------------------------------------------------------ 2. page ring
__global__ void pages_kernel(epa::Arena a, int rounds, int n, int* sink) {
    for (int r = 0; r < rounds; ++r) {
        const long long t = a.warp_try_take(n);
        if (t < 0) { if ((threadIdx.x & 31) == 0) atomicAdd(sink, 1); continue; }
        int pages[8];
        for (int i = 0; i < n; ++i) pages[i] = a.page_at(static_cast<unsigned long long>(t), i);
        a.give(pages, n);
    }
}

// ------------------------------------------------------------------ 3. variable output
// Thread t emits n(t) records of 64 B; n is skewed (most threads few, a few many).
// Learning n(t) costs `work` dependent hash rounds (a stand-in for the real work that
// decides how much a thread outputs: a filter, a join probe, a BFS expansion).
__device__ __forceinline__ int n_out(int t, int work) {
    unsigned h = hash(t);
    for (int i = 0; i < work; ++i) h = hash(h + i);
    h = hash(t) ^ (h & 0x80000000u);   // same distribution for every `work`
    return (h % 100 < 2) ? 64 + h % 64 : h % 8;   // mean about 5.6, max 127
}
__device__ __forceinline__ void emit(int4* rec, int t, int i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) rec[j] = make_int4(t, i, j, t ^ i);
}
constexpr int kMaxOut = 128;
__global__ void out_worst_case(int4* out, int n_threads, int work) {     // [n_threads][kMaxOut] records
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_threads) return;
    const int n = n_out(t, work);
    for (int i = 0; i < n; ++i) emit(out + (int64_t(t) * kMaxOut + i) * 4, t, i);
}
__global__ void out_count(int* counts, int n_threads, int work) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n_threads) counts[t] = n_out(t, work);
}
__global__ void out_write(int4* out, const int* offsets, int n_threads, int work) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_threads) return;
    const int n = n_out(t, work);   // the second pass redoes the work
    for (int i = 0; i < n; ++i) emit(out + (int64_t(offsets[t]) + i) * 4, t, i);
}
__global__ void out_arena(epa::PagedArray pa, long long* first, int n_threads, int work) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_threads) return;
    const int n = n_out(t, work);
    // warp-aggregated append: one fetch-add per warp
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
    int prefix = 0, total = 0;
    for (unsigned m = mask; m; m &= m - 1) {
        const int l = __ffs(m) - 1, v = __shfl_sync(mask, n, l);
        if (l < lane) prefix += v;
        total += v;
    }
    long long r = 0;
    if (lane == leader) r = pa.append(total);
    r = __shfl_sync(mask, r, leader) + prefix;
    first[t] = r;
    // resolve the page once per page, then write rows contiguously inside it
    char* row = nullptr;
    for (int i = 0; i < n; ++i) {
        if (row == nullptr or (r + i) % pa.rows_per_page == 0) row = pa.row(r + i);
        else row += pa.row_bytes;
        emit(reinterpret_cast<int4*>(row), t, i);
    }
}
__global__ void release_kernel(epa::PagedArray pa) { pa.release(); }

int main() {
    int ndev;
    CHECK(cudaGetDeviceCount(&ndev));
    CHECK(cudaSetDevice(0));
    int* sink;
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemset(sink, 0, 4));
    const int blocks = 148 * 4, threads = 256, rounds = 64;

    printf("1. malloc/free, %d threads x %d rounds, 4 live objects per thread\n", blocks * threads, rounds);
    CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t(2) << 30));
    for (int size : {16, 64, 256, 1024}) {
        auto a = make_arena(0, 32768, 64 * 1024);   // 2 GiB of 64 KiB pages
        const double ops = double(blocks) * threads * rounds;
        const float t_cuda = time_ms([&] { malloc_kernel<false><<<blocks, threads>>>(a, size, rounds, sink); }, 3);
        const float t_epa = time_ms([&] { malloc_kernel<true><<<blocks, threads>>>(a, size, rounds, sink); }, 3);
        epa::ArenaMeta m;
        CHECK(cudaMemcpy(&m, a.base, sizeof m, cudaMemcpyDeviceToHost));
        printf("   %5d B   CUDA malloc %8.1f M/s   epa::WarpHeap %8.1f M/s   (%.0fx)   pages back: %s\n", size,
               ops / t_cuda / 1e3, ops / t_epa / 1e3, t_cuda / t_epa, m.free_pages == a.num_pages ? "all" : "NO");
        CHECK(cudaFree(a.base));
    }
    int fails;
    CHECK(cudaMemcpy(&fails, sink, 4, cudaMemcpyDeviceToHost));
    printf("   failed allocations: %d\n", fails);

    printf("2. page take + give, one warp-aggregated take per warp per round\n");
    for (int n : {1, 4}) {
        auto a = make_arena(0, blocks * threads * n + 1024, 4096);   // room for every lane: failures are real shortage only
        const double pages = double(blocks) * threads * rounds * n;
        const float t = time_ms([&] { pages_kernel<<<blocks, threads>>>(a, rounds, n, sink); }, 3);
        int f;
        CHECK(cudaMemcpy(&f, sink, 4, cudaMemcpyDeviceToHost));
        printf("   local, %d page(s) per lane:   %8.1f M pages/s   (failed warp takes so far: %d)\n", n, pages / t / 1e3, f);
        if (ndev >= 2) {
            int can;
            CHECK(cudaDeviceCanAccessPeer(&can, 1, 0));
            if (can) {
                CHECK(cudaSetDevice(1));
                cudaDeviceEnablePeerAccess(0, 0);
                cudaGetLastError();
                int* sink1;
                CHECK(cudaMalloc(&sink1, 4));
                const float tr = time_ms([&] { pages_kernel<<<blocks, threads>>>(a, rounds / 8, n, sink1); }, 3);
                printf("   GPU 1 -> GPU 0's arena (NVLink), %d page(s): %8.1f M pages/s\n", n, pages / 8 / tr / 1e3);
                CHECK(cudaSetDevice(0));
            }
        }
        CHECK(cudaFree(a.base));
    }

    printf("3. variable-size output (64 B records, skewed counts per thread)\n");
    for (int work : {0, 256})
    for (int n_threads : {1 << 20, 1 << 23}) {
        const int nb = (n_threads + 255) / 256;
        // worst case
        int4* worst;
        const size_t worst_bytes = size_t(n_threads) * kMaxOut * 64;
        CHECK(cudaMalloc(&worst, worst_bytes));
        const float t_worst = time_ms([&] { out_worst_case<<<nb, 256>>>(worst, n_threads, work); });
        CHECK(cudaFree(worst));
        // count + scan + write (exact, two passes)
        int *counts, *offsets;
        CHECK(cudaMalloc(&counts, n_threads * 4));
        CHECK(cudaMalloc(&offsets, n_threads * 4));
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, counts, offsets, n_threads);
        void* tmp;
        CHECK(cudaMalloc(&tmp, tmp_bytes));
        out_count<<<nb, 256>>>(counts, n_threads, work);
        std::vector<int> hc(n_threads);
        CHECK(cudaMemcpy(hc.data(), counts, n_threads * 4, cudaMemcpyDeviceToHost));
        long long total = 0;
        for (int c : hc) total += c;
        int4* exact;
        CHECK(cudaMalloc(&exact, total * 64));
        const float t_scan = time_ms([&] {
            out_count<<<nb, 256>>>(counts, n_threads, work);
            cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, counts, offsets, n_threads);
            out_write<<<nb, 256>>>(exact, offsets, n_threads, work);
        });
        // PagedArray, one pass; the arena is shared and could be sized for any mix
        const int64_t page_bytes = 64 * 1024, rpp = page_bytes / 64;
        const int max_pages = static_cast<int>((total + rpp - 1) / rpp) + 1024;
        auto a = make_arena(0, max_pages, page_bytes);
        epa::PagedArray pa;
        pa.arena = a, pa.row_bytes = 64, pa.rows_per_page = rpp;
        pa.max_pages = static_cast<int>(int64_t(n_threads) * kMaxOut / rpp + 1);
        CHECK(cudaMalloc(&pa.table, int64_t(pa.max_pages) * 4));
        CHECK(cudaMemset(pa.table, 0xff, int64_t(pa.max_pages) * 4));
        CHECK(cudaMalloc(&pa.length, 8));
        CHECK(cudaMemset(pa.length, 0, 8));
        long long* first;
        CHECK(cudaMalloc(&first, int64_t(n_threads) * 8));
        const float t_arena = time_ms([&] {
            out_arena<<<nb, 256>>>(pa, first, n_threads, work);
            release_kernel<<<1, 1024>>>(pa);
        });
        out_arena<<<nb, 256>>>(pa, first, n_threads, work);
        epa::ArenaMeta m;
        CHECK(cudaMemcpy(&m, a.base, sizeof m, cudaMemcpyDeviceToHost));
        const double used = double(a.num_pages - m.free_pages) * page_bytes;
        release_kernel<<<1, 1024>>>(pa);
        CHECK(cudaDeviceSynchronize());
        printf("   %8d threads, work %d, %lld records (%.1f MiB):\n", n_threads, work, total, total * 64 / 1048576.0);
        printf("      worst case [T][%d]    %8.1f MiB  %7.3f ms\n", kMaxOut, worst_bytes / 1048576.0, t_worst);
        printf("      count + scan + write  %8.1f MiB  %7.3f ms\n", total * 64 / 1048576.0, t_scan);
        printf("      PagedArray append     %8.1f MiB  %7.3f ms  (pages in use; release included)\n",
               used / 1048576.0, t_arena);
        CHECK(cudaFree(counts));
        CHECK(cudaFree(offsets));
        CHECK(cudaFree(tmp));
        CHECK(cudaFree(exact));
        CHECK(cudaFree(a.base));
        CHECK(cudaFree(pa.table));
        CHECK(cudaFree(first));
    }
    return 0;
}
