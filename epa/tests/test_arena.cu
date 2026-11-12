// Stress tests of epa::Arena, WarpHeap and PagedArray (one GPU, plus a peer GPU when
// two are visible). Every test ends by checking that every page came back.
// Build: nvcc -std=c++17 -arch=sm_100a -I ../include test_arena.cu -o test_arena
#include <cstdio>
#include <cstring>
#include <vector>

#include <epa/arena.cuh>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ unsigned hash(unsigned x) {
    x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
    return x;
}

struct Region {
    epa::Arena a;
    char* mem;
};
Region make_arena(int num_pages, int64_t page_bytes) {
    Region r;
    const auto bytes = epa::Arena::bytes_for(num_pages, page_bytes);
    CHECK(cudaMalloc(&r.mem, bytes));
    r.a.base = r.mem, r.a.page_bytes = page_bytes, r.a.num_pages = num_pages;
    epa::arena_init_kernel<<<64, 256>>>(r.a);
    CHECK(cudaDeviceSynchronize());
    return r;
}
epa::ArenaMeta meta(const epa::Arena& a) {
    epa::ArenaMeta m;
    CHECK(cudaMemcpy(&m, a.base, sizeof(m), cudaMemcpyDeviceToHost));
    return m;
}
// Put an arena's ring in the state it has after `t0` takes and gives (all pages free):
// head = t0, tail = t0 + N, and each slot holds the page of its next take ticket.
void init_at_ticket(const epa::Arena& a, unsigned long long t0) {
    const int n = a.num_pages;
    std::vector<char> img(epa::Arena::meta_bytes(n), 0);
    epa::ArenaMeta m{};
    m.head = t0, m.tail = t0 + n, m.free_pages = n, m.min_free_pages = n;
    std::memcpy(img.data(), &m, sizeof m);
    auto* slots = reinterpret_cast<unsigned long long*>(img.data() + 256);
    for (int i = 0; i < n; ++i) {
        const unsigned long long t = t0 + i;
        slots[t % n] = (epa::Arena::seq_of(t + 1) << epa::Arena::kIdBits) | static_cast<unsigned>(i);
    }
    CHECK(cudaMemcpy(a.base, img.data(), img.size(), cudaMemcpyHostToDevice));
}
int g_fail = 0;
void expect(bool ok, const char* what) {
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += not ok;
}

// 1. page ring: every warp takes 1..8 pages, checks it holds them alone, gives them back
__global__ void ring_kernel(epa::Arena a, int* owner, int iters, int* errors) {
    const int lane = threadIdx.x & 31, wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    for (int it = 0; it < iters; ++it) {
        const int n = 1 + hash(wid * 977 + it) % 8;
        long long t = -1;
        if (lane == 0) t = a.try_take(n);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t < 0) continue;
        int pages[8];
        if (lane == 0) {
            for (int i = 0; i < n; ++i) {
                pages[i] = a.page_at(static_cast<unsigned long long>(t), i);
                if (atomicExch(owner + pages[i], wid) != -1) atomicAdd(errors, 1);
            }
            for (int i = 0; i < n; ++i)
                if (atomicExch(owner + pages[i], -1) != wid) atomicAdd(errors, 1);
            a.give(pages, n);
        }
        __syncwarp();
    }
}

// 2. WarpHeap: allocate, write a signature, hand pointers to another block for checking
__global__ void heap_alloc_kernel(epa::Arena a, void** ptrs, int* sizes, int per_thread, int round) {
    epa::WarpHeap h{a};
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < per_thread; ++i) {
        const int sz = 16 + hash(tid * 131 + i + round * 7919) % 2033;   // 16 .. 2048 B
        const int want = (hash(tid + i * 3) % 5 == 0) ? 0 : sz;          // some lanes skip
        auto* p = static_cast<unsigned*>(h.alloc(want));
        const int idx = tid * per_thread + i;
        ptrs[idx] = p, sizes[idx] = want;
        if (p != nullptr)
            for (int j = 0; j < want / 4; ++j) p[j] = idx * 2654435761u + j;
    }
    h.retire();
}
__global__ void heap_check_free_kernel(epa::Arena a, void** ptrs, const int* sizes, int n, int* errors) {
    // permuted order, so frees come from different SMs than allocations
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
        const int idx = static_cast<int>((static_cast<long long>(k) * 7919) % n);
        auto* p = static_cast<unsigned*>(ptrs[idx]);
        if (p == nullptr) { if (sizes[idx] != 0) atomicAdd(errors, 1); continue; }
        for (int j = 0; j < sizes[idx] / 4; ++j)
            if (p[j] != idx * 2654435761u + j) { atomicAdd(errors, 1); break; }
        epa::WarpHeap::free(a, p);
    }
}

// 3. PagedArray: threads append 1..16 rows, write them, verify, release
__global__ void paged_append_kernel(epa::PagedArray pa, long long* first, int* count, int* errors) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = 1 + hash(tid) % 16;
    const long long r = pa.append(n);
    first[tid] = r, count[tid] = n;
    for (int i = 0; i < n; ++i) {
        auto* row = reinterpret_cast<long long*>(pa.row(r + i));
        if (row == nullptr) continue;   // page could not be taken (arena empty)
        for (int j = 0; j < pa.row_bytes / 8; ++j) row[j] = (static_cast<long long>(tid) << 20) + i * 64 + j;
    }
}
__global__ void paged_check_kernel(epa::PagedArray pa, const long long* first, const int* count, int n_threads,
                                   int* errors, int* placed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    for (int i = 0; i < count[tid]; ++i) {
        auto* row = reinterpret_cast<long long*>(pa.row(first[tid] + i));
        if (row == nullptr) continue;
        atomicAdd(placed, 1);
        for (int j = 0; j < pa.row_bytes / 8; ++j)
            if (row[j] != (static_cast<long long>(tid) << 20) + i * 64 + j) { atomicAdd(errors, 1); break; }
    }
}
__global__ void paged_release_kernel(epa::PagedArray pa) { pa.release(); }

int run_paged(int num_pages, int threads, const char* label) {
    auto R = make_arena(num_pages, 8 * 256);   // 8 rows of 256 B per page
    epa::PagedArray pa;
    pa.arena = R.a, pa.row_bytes = 256, pa.rows_per_page = 8, pa.max_pages = threads * 16 / 8 + 2;
    CHECK(cudaMalloc(&pa.table, pa.max_pages * 4));
    CHECK(cudaMemset(pa.table, 0xff, pa.max_pages * 4));
    CHECK(cudaMalloc(&pa.length, 8));
    CHECK(cudaMemset(pa.length, 0, 8));
    long long* first;
    int *count, *errors, *placed;
    CHECK(cudaMalloc(&first, threads * 8)); CHECK(cudaMalloc(&count, threads * 4));
    CHECK(cudaMalloc(&errors, 4)); CHECK(cudaMalloc(&placed, 4));
    CHECK(cudaMemset(errors, 0, 4)); CHECK(cudaMemset(placed, 0, 4));
    paged_append_kernel<<<threads / 256, 256>>>(pa, first, count, errors);
    paged_check_kernel<<<threads / 256, 256>>>(pa, first, count, threads, errors, placed);
    CHECK(cudaDeviceSynchronize());
    int e, pl;
    unsigned long long len;
    CHECK(cudaMemcpy(&e, errors, 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&pl, placed, 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&len, pa.length, 8, cudaMemcpyDeviceToHost));
    const auto m = meta(R.a);
    const long long used_pages = num_pages - m.free_pages;
    paged_release_kernel<<<1, 256>>>(pa);
    CHECK(cudaDeviceSynchronize());
    char buf[200];
    snprintf(buf, sizeof buf, "%s: %llu rows, %d placed, %lld pages used", label, len, pl, used_pages);
    // enough pages: every row placed; too few: exactly the rows of the pages taken
    // (the last page of the array may be partly filled)
    const bool rows_ok = num_pages >= (long long)(len + 7) / 8 ? pl == (long long)len
                                                               : pl <= used_pages * 8 and pl >= used_pages * 8 - 7;
    expect(e == 0 and meta(R.a).free_pages == num_pages and rows_ok, buf);
    return 0;
}

int main() {
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    CHECK(cudaSetDevice(0));
    printf("epa::Arena tests\n");

    {   // 1
        const int N = 1024;
        auto R = make_arena(N, 4096);
        int *owner, *errors;
        CHECK(cudaMalloc(&owner, N * 4)); CHECK(cudaMemset(owner, 0xff, N * 4));
        CHECK(cudaMalloc(&errors, 4)); CHECK(cudaMemset(errors, 0, 4));
        ring_kernel<<<264, 256>>>(R.a, owner, 200, errors);   // 2112 warps on 1024 pages
        CHECK(cudaDeviceSynchronize());
        int e;
        CHECK(cudaMemcpy(&e, errors, 4, cudaMemcpyDeviceToHost));
        const auto m = meta(R.a);
        char buf[200];
        snprintf(buf, sizeof buf, "page ring: %llu pages taken, %llu failed takes, exclusive", m.taken, m.failed);
        expect(e == 0 and m.free_pages == N and m.head + N == m.tail, buf);
    }
    {   // 1b. tickets past 2^40, where a slot's 40-bit seq wraps (it used to hang there)
        const int N = 1024;
        const unsigned long long wrap = 1ull << (64 - epa::Arena::kIdBits);
        auto R = make_arena(N, 4096);
        init_at_ticket(R.a, wrap - 4 * N);   // 4 laps below; the run takes ~17K pages
        int *owner, *errors;
        CHECK(cudaMalloc(&owner, N * 4)); CHECK(cudaMemset(owner, 0xff, N * 4));
        CHECK(cudaMalloc(&errors, 4)); CHECK(cudaMemset(errors, 0, 4));
        ring_kernel<<<264, 256>>>(R.a, owner, 200, errors);
        CHECK(cudaDeviceSynchronize());
        int e;
        CHECK(cudaMemcpy(&e, errors, 4, cudaMemcpyDeviceToHost));
        const auto m = meta(R.a);
        char buf[200];
        snprintf(buf, sizeof buf, "page ring across ticket 2^40: head 2^40 + %lld, exclusive",
                 static_cast<long long>(m.head - wrap));
        expect(e == 0 and m.free_pages == N and m.head + N == m.tail and m.head > wrap, buf);
    }
    {   // 2
        const int N = 8192, threads = 148 * 256, per = 8;   // 512 MiB, ~310 MB live
        auto R = make_arena(N, 64 * 1024);
        void** ptrs;
        int *sizes, *errors;
        CHECK(cudaMalloc(&ptrs, threads * per * 8)); CHECK(cudaMalloc(&sizes, threads * per * 4));
        CHECK(cudaMalloc(&errors, 4)); CHECK(cudaMemset(errors, 0, 4));
        long long objects = 0;
        for (int round = 0; round < 5; ++round) {
            heap_alloc_kernel<<<threads / 256, 256>>>(R.a, ptrs, sizes, per, round);
            heap_check_free_kernel<<<148, 256>>>(R.a, ptrs, sizes, threads * per, errors);
            CHECK(cudaDeviceSynchronize());
            objects += threads * per;
        }
        int e;
        CHECK(cudaMemcpy(&e, errors, 4, cudaMemcpyDeviceToHost));
        const auto m = meta(R.a);
        char buf[200];
        snprintf(buf, sizeof buf, "WarpHeap: %lld malloc/free, %d errors, low water %d of %d pages", objects, e, m.min_free_pages, N);
        expect(e == 0 and m.free_pages == N, buf);
    }
    run_paged(80000, 64 * 1024, "PagedArray, enough pages");
    run_paged(3000, 64 * 1024, "PagedArray, arena too small (try policy)");

    if (ndev >= 2) {   // 4. a peer GPU allocates from GPU 0's arena over NVLink and GPU 0 frees
        int can = 0;
        CHECK(cudaDeviceCanAccessPeer(&can, 1, 0));
        if (can) {
            CHECK(cudaSetDevice(0));
            const int N = 8192, threads = 148 * 256, per = 4;
            auto R = make_arena(N, 64 * 1024);
            CHECK(cudaSetDevice(1));
            CHECK(cudaDeviceEnablePeerAccess(0, 0));
            void** ptrs;
            int *sizes, *errors;
            CHECK(cudaMallocManaged(&ptrs, threads * per * 8)); CHECK(cudaMallocManaged(&sizes, threads * per * 4));
            CHECK(cudaMallocManaged(&errors, 4));
            *errors = 0;
            heap_alloc_kernel<<<threads / 256, 256>>>(R.a, ptrs, sizes, per, 0);   // runs on GPU 1
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaSetDevice(0));
            heap_check_free_kernel<<<148, 256>>>(R.a, ptrs, sizes, threads * per, errors);   // GPU 0 frees
            CHECK(cudaDeviceSynchronize());
            const auto m = meta(R.a);
            char buf[200];
            snprintf(buf, sizeof buf, "remote: GPU 1 mallocs in GPU 0's arena, GPU 0 frees (%d errors, peak %d pages)", *errors, N - m.min_free_pages);
            expect(*errors == 0 and m.free_pages == N, buf);
        }
    }
    printf(g_fail ? "FAILED (%d)\n" : "all arena tests passed\n", g_fail);
    return g_fail != 0;
}
