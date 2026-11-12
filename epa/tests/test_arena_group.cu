// epa::ArenaGroup: allocation falls back to peer GPUs' arenas when the local one is short.
// Build: nvcc -std=c++17 -O3 -arch=sm_100a -I ../include test_arena_group.cu -o test_arena_group
// Run:   ./test_arena_group      (needs >= 2 GPUs with peer access; uses up to 4)
//
// GPU 0 has a small arena, the peers larger ones. Checks:
//   1. PagedArray on GPU 0 that outgrows its arena: rows land on peers, data reads back,
//      release returns every page to its owner
//   2. peer_floor: peers lend only above it; rows beyond are "not placed" (try policy)
//   3. WarpHeap on GPU 0 past its arena, objects freed by GPU 1 (owner found by address)
//   4. latency of one take + give: local, and through the fallback on a peer
#include <epa/arena.cuh>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static int g_fail = 0;
static void expect(bool ok, const char* what) { printf("  [%s] %s\n", ok ? "ok" : "FAIL", what); g_fail += !ok; }

struct Gpu { int dev; epa::Arena a; void* base; };

static epa::ArenaMeta meta(const Gpu& g) {
    epa::ArenaMeta m;
    CHECK(cudaSetDevice(g.dev));
    CHECK(cudaMemcpy(&m, g.a.meta(), sizeof m, cudaMemcpyDeviceToHost));
    return m;
}
static void reset(Gpu& g) {
    CHECK(cudaSetDevice(g.dev));
    const auto img = epa::Arena::host_init_image(g.a.num_pages);
    CHECK(cudaMemcpy(g.a.base, img.data(), img.size(), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------ kernels
__global__ void paged_append_kernel(epa::PagedArray pa, int n_rows, int per_call) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i * per_call < n_rows; i += gridDim.x * blockDim.x) {
        const int n = min(per_call, n_rows - i * per_call);
        const long long r = pa.append(n);
        for (int j = 0; j < n; ++j) {
            char* p = pa.row(r + j);
            if (p != nullptr)
                for (int64_t b = 0; b < pa.row_bytes; b += 8) *reinterpret_cast<long long*>(p + b) = (r + j) * 1000003ll + b;
        }
    }
}
__global__ void paged_check_kernel(epa::PagedArray pa, int* errors, int* placed, int* per_owner) {
    const long long n = static_cast<long long>(*pa.length);
    for (long long r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        const char* p = pa.row(r);
        if (p == nullptr) continue;
        atomicAdd(placed, 1);
        for (int64_t b = 0; b < pa.row_bytes; b += 8)
            if (*reinterpret_cast<const long long*>(p + b) != r * 1000003ll + b) { atomicAdd(errors, 1); break; }
    }
    const int pages = static_cast<int>((n + pa.rows_per_page - 1) / pa.rows_per_page);
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < pages and q < pa.max_pages; q += gridDim.x * blockDim.x)
        if (pa.table[q] >= 0) atomicAdd(per_owner + epa::ArenaGroup::owner_of_code(pa.table[q]), 1);
}
__global__ void paged_release_kernel(epa::PagedArray pa) { pa.release(); }

__global__ void heap_alloc_kernel(epa::Arena a, epa::ArenaGroup g, void** ptrs, int per, int* nulls) {
    epa::WarpHeap h{a};
    h.group = g;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < per; ++k) {
        const int bytes = 64 + ((tid * 7 + k * 13) % 16) * 64;
        void* p = h.alloc(bytes);
        ptrs[tid * per + k] = p;
        if (p == nullptr) atomicAdd(nulls, 1);
        else for (int b = 0; b < bytes; b += 4) reinterpret_cast<int*>(p)[b / 4] = tid * per + k;
    }
    h.retire();
}
__global__ void heap_free_kernel(epa::ArenaGroup g, void** ptrs, int n, int* errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        void* p = ptrs[i];
        if (p == nullptr) continue;
        if (*reinterpret_cast<int*>(p) != i or g.owner_of(p) < 0) atomicAdd(errors, 1);
        epa::WarpHeap::free(g, p);
    }
}

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// One thread: iters x (take 1 through the group, read its id, give it back); ns per step
__global__ void latency_kernel(epa::ArenaGroup g, int iters, unsigned long long* out, int* owner_seen) {
    unsigned long long t_take = 0, t_give = 0;
    int owner = -1;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = gtime();
        const long long t = g.try_take(1, &owner);
        if (t < 0) { out[0] = ~0ull; return; }
        const int c = g.page_at(owner, static_cast<unsigned long long>(t), 0);
        const auto t1 = gtime();
        g.give(c);
        const auto t2 = gtime();
        t_take += t1 - t0, t_give += t2 - t1;
    }
    out[0] = t_take / iters, out[1] = t_give / iters;
    *owner_seen = owner;
}

int main() {
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    const int R = ndev < 4 ? ndev : 4;
    if (R < 2) { printf("needs >= 2 GPUs\n"); return 0; }
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        for (int p = 0; p < R; ++p) {
            int can = 0;
            if (p == d) continue;
            CHECK(cudaDeviceCanAccessPeer(&can, d, p));
            if (!can) { printf("no peer access %d -> %d\n", d, p); return 0; }
            cudaDeviceEnablePeerAccess(p, 0);
        }
    }
    printf("epa::ArenaGroup tests (%d GPUs)\n", R);

    // Arenas: GPU 0 small, peers 128 pages; same layout so a page id means the same offset
    const int64_t page_bytes = 64 * 1024;
    const int pages_small = 8, pages_big = 128;
    std::vector<Gpu> gpus(R);
    std::vector<void*> bases(R);
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        const int n = d == 0 ? pages_small : pages_big;
        // allocate the largest size on every GPU, so the arena sits at offset 0 of each base
        CHECK(cudaMalloc(&gpus[d].base, epa::Arena::bytes_for(pages_big, page_bytes)));
        gpus[d].dev = d;
        gpus[d].a.base = static_cast<char*>(gpus[d].base);
        gpus[d].a.page_bytes = page_bytes;
        gpus[d].a.num_pages = n;
        bases[d] = gpus[d].base;
        reset(gpus[d]);
    }
    // NOTE: page_ptr / owns use the arena's own num_pages; a peer view copies the local
    // Arena (num_pages of GPU 0), so peers are viewed with *their* sizes via per-rank arenas.
    // ArenaGroup::arena(d) = local.on(pt, d) keeps the local num_pages, so all arenas must
    // have the same page count: use pages_big everywhere and shrink GPU 0 by taking pages.
    for (int d = 0; d < R; ++d) { gpus[d].a.num_pages = pages_big; reset(gpus[d]); }

    // Peer tables, one device copy per GPU
    std::vector<void**> pt_dev(R);
    for (int d = 0; d < R; ++d) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaMalloc(&pt_dev[d], R * sizeof(void*)));
        CHECK(cudaMemcpy(pt_dev[d], bases.data(), R * sizeof(void*), cudaMemcpyHostToDevice));
    }
    auto group_on = [&](int d, int floor) {
        epa::ArenaGroup g;
        g.local = gpus[d].a, g.pt = {bases[d], pt_dev[d]}, g.rank = d, g.num_ranks = R, g.peer_floor = floor;
        return g;
    };
    // GPU 0 keeps only `pages_small` free: hold the rest with a host-side ring image
    auto shrink0 = [&]() {
        CHECK(cudaSetDevice(0));
        auto img = epa::Arena::host_init_image(pages_big);
        auto* m = reinterpret_cast<epa::ArenaMeta*>(img.data());
        m->free_pages = pages_small, m->min_free_pages = pages_small;
        m->tail = pages_small;   // tickets [0, pages_small) hold pages 0..7; the rest are "held"
        auto* slots = reinterpret_cast<unsigned long long*>(img.data() + 256);
        for (int i = pages_small; i < pages_big; ++i)
            slots[i] = static_cast<unsigned long long>(i) << epa::Arena::kIdBits;   // empty slot waiting for give ticket i
        CHECK(cudaMemcpy(gpus[0].a.base, img.data(), img.size(), cudaMemcpyHostToDevice));
    };
    auto reset_all = [&]() { for (int d = 0; d < R; ++d) reset(gpus[d]); shrink0(); };
    auto all_free = [&]() {
        bool ok = meta(gpus[0]).free_pages == pages_small;
        for (int d = 1; d < R; ++d) ok = ok and meta(gpus[d]).free_pages == pages_big;
        return ok;
    };

    int *errors, *placed, *per_owner, *nulls;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMallocManaged(&errors, 4)); CHECK(cudaMallocManaged(&placed, 4));
    CHECK(cudaMallocManaged(&per_owner, 4 * 8)); CHECK(cudaMallocManaged(&nulls, 4));

    auto run_paged = [&](int n_rows, int floor, const char* label) {
        reset_all();
        CHECK(cudaSetDevice(0));
        const int rows_per_page = 64, max_pages = 512;
        const int64_t row_bytes = page_bytes / rows_per_page;
        int* table; unsigned long long* length;
        CHECK(cudaMalloc(&table, max_pages * 4)); CHECK(cudaMemset(table, 0xff, max_pages * 4));
        CHECK(cudaMalloc(&length, 8)); CHECK(cudaMemset(length, 0, 8));
        epa::PagedArray pa{gpus[0].a, table, length, row_bytes, rows_per_page, max_pages};
        pa.group = group_on(0, floor);
        *errors = *placed = 0;
        for (int d = 0; d < 8; ++d) per_owner[d] = 0;
        paged_append_kernel<<<64, 128>>>(pa, n_rows, 3);
        paged_check_kernel<<<64, 256>>>(pa, errors, placed, per_owner);
        CHECK(cudaDeviceSynchronize());
        const auto b0 = meta(gpus[0]).borrowed;
        paged_release_kernel<<<1, 256>>>(pa);
        CHECK(cudaDeviceSynchronize());
        char buf[300];
        snprintf(buf, sizeof buf, "%s: %d rows, %d placed, pages on GPU 0..%d = %d %d %d %d, borrowed %llu, %d bad rows",
                 label, n_rows, *placed, R - 1, per_owner[0], per_owner[1], R > 2 ? per_owner[2] : 0, R > 3 ? per_owner[3] : 0,
                 b0, *errors);
        const int expect_pages = (n_rows + rows_per_page - 1) / rows_per_page;
        const int cap = pages_small + (R - 1) * (pages_big - floor);
        const bool placed_ok = expect_pages <= cap ? *placed == n_rows : *placed < n_rows and *placed > 0;
        expect(*errors == 0 and placed_ok and per_owner[0] <= pages_small and all_free(), buf);
        CHECK(cudaFree(table)); CHECK(cudaFree(length));
    };
    run_paged(64 * 300, 0, "1. PagedArray past GPU 0's 8 pages");
    run_paged(64 * 300, 100, "2. peer_floor = 100 of 128 (peers lend 28 each)");

    {   // 3. WarpHeap on GPU 0 past its arena; GPU 1 frees
        reset_all();
        CHECK(cudaSetDevice(0));
        const int threads = 32 * 64, per = 16, n = threads * per;   // ~10 MB of objects, 8 local pages = 0.5 MB
        void** ptrs;
        CHECK(cudaMallocManaged(&ptrs, n * 8));
        *nulls = 0, *errors = 0;
        heap_alloc_kernel<<<threads / 128, 128>>>(gpus[0].a, group_on(0, 0), ptrs, per, nulls);
        CHECK(cudaDeviceSynchronize());
        const auto b0 = meta(gpus[0]).borrowed;
        CHECK(cudaSetDevice(1));
        heap_free_kernel<<<64, 256>>>(group_on(1, 0), ptrs, n, errors);   // GPU 1 frees GPU 0's objects
        CHECK(cudaDeviceSynchronize());
        char buf[300];
        snprintf(buf, sizeof buf, "3. WarpHeap on GPU 0, %d objects (%d null), borrowed %llu pages, GPU 1 frees, %d errors",
                 n, *nulls, b0, *errors);
        expect(*nulls == 0 and *errors == 0 and b0 > 0 and all_free(), buf);
        CHECK(cudaFree(ptrs));
    }

    {   // 4. latency
        unsigned long long* out;
        int* owner;
        CHECK(cudaSetDevice(0));
        CHECK(cudaMallocManaged(&out, 16)); CHECK(cudaMallocManaged(&owner, 4));
        for (int remote = 0; remote < 2; ++remote) {
            reset_all();
            CHECK(cudaSetDevice(0));
            if (remote) {   // GPU 0's arena empty: every take falls back to GPU 1
                auto img = epa::Arena::host_init_image(pages_big);
                auto* m = reinterpret_cast<epa::ArenaMeta*>(img.data());
                m->free_pages = 0, m->tail = 0;
                auto* slots = reinterpret_cast<unsigned long long*>(img.data() + 256);
                for (int i = 0; i < pages_big; ++i) slots[i] = static_cast<unsigned long long>(i) << epa::Arena::kIdBits;
                CHECK(cudaMemcpy(gpus[0].a.base, img.data(), img.size(), cudaMemcpyHostToDevice));
            }
            latency_kernel<<<1, 1>>>(group_on(0, 0), 2000, out, owner);
            CHECK(cudaDeviceSynchronize());
            printf("  latency, %s: take + page_at %llu ns, give %llu ns (pages from GPU %d)\n",
                   remote ? "local empty, fallback to a peer" : "local arena", out[0], out[1], *owner);
        }
        CHECK(cudaFree(out)); CHECK(cudaFree(owner));
    }
    printf(g_fail ? "FAILED (%d)\n" : "all arena group tests passed\n", g_fail);
    return g_fail != 0;
}
