// epa host tier: the last fallback (HostArena) and PagedWeights.
// Build: nvcc -std=c++17 -O3 -arch=sm_100a -I ../include test_host_tier.cu -lcuda -o test_host_tier
// Run:   ./test_host_tier          (needs 2 GPUs with peer access for test 1)
//   1. ArenaGroup on GPU 0 takes 40 pages one by one: 8 local, 8 from GPU 1, the rest from host
//      memory (only once both GPUs are out); every page holds its data; all go back to their rings
//   2. PagedWeights, 4 layers of 64 MB with an 8 MB shared tail: after page_in(k) every byte of
//      layer k's contiguous view is right; HBM held = 4 bodies + 1 tail
#include <epa/host_tier.hpp>

#include <cstdio>
#include <vector>

static int g_fail = 0;
static void expect(bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what); g_fail += !ok; }

__global__ void take_kernel(epa::ArenaGroup g, int n, int* codes, int* per_owner, int* bad) {
    for (int i = 0; i < n; ++i) {
        int owner;
        const long long t = g.try_take(1, &owner);
        if (t < 0) { codes[i] = -1; continue; }
        const int c = g.page_at(owner, static_cast<unsigned long long>(t), 0);
        codes[i] = c;
        atomicAdd(per_owner + (owner == epa::ArenaGroup::kHost ? 2 : owner), 1);
        int* p = reinterpret_cast<int*>(g.ptr(c));
        for (int k = 0; k < 1024; ++k) p[k] = i * 131 + k;
    }
    for (int i = 0; i < n; ++i) {
        if (codes[i] < 0) continue;
        const int* p = reinterpret_cast<const int*>(g.ptr(codes[i]));
        for (int k = 0; k < 1024; ++k) if (p[k] != i * 131 + k) { atomicAdd(bad, 1); break; }
        g.give(codes[i]);
    }
}
__global__ void fill_kernel(char* p, int64_t n, int salt) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        p[i] = static_cast<char>((i * 31 + salt * 7) & 0xff);
}
__global__ void check_kernel(const char* p, int64_t n, int salt, unsigned long long* bad) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        if (p[i] != static_cast<char>((i * 31 + salt * 7) & 0xff)) atomicAdd(bad, 1ull);
}

int main() {
    std::printf("epa host tier tests\n");
    int ndev = 0;
    EPA_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev >= 2) {   // 1. fallback order
        const int64_t pb = 64 * 1024;
        const int n_gpu = 8, n_host = 64, want = 40;
        void* base[2];
        for (int d = 0; d < 2; ++d) {
            EPA_CUDA_CHECK(cudaSetDevice(d));
            EPA_CUDA_CHECK(cudaMalloc(&base[d], epa::Arena::bytes_for(n_gpu, pb)));
            const auto img = epa::Arena::host_init_image(n_gpu);
            EPA_CUDA_CHECK(cudaMemcpy(base[d], img.data(), img.size(), cudaMemcpyHostToDevice));
            cudaDeviceEnablePeerAccess(1 - d, 0);
        }
        cudaGetLastError();
        EPA_CUDA_CHECK(cudaSetDevice(0));
        void** pt;
        EPA_CUDA_CHECK(cudaMalloc(&pt, 2 * sizeof(void*)));
        EPA_CUDA_CHECK(cudaMemcpy(pt, base, 2 * sizeof(void*), cudaMemcpyHostToDevice));
        auto host = epa::HostArena::create(n_host, pb);
        epa::ArenaGroup g;
        g.local.base = static_cast<char*>(base[0]), g.local.page_bytes = pb, g.local.num_pages = n_gpu;
        g.pt = {base[0], pt}, g.rank = 0, g.num_ranks = 2, g.host = host.arena;
        int *codes, *per_owner, *bad;
        EPA_CUDA_CHECK(cudaMallocManaged(&codes, want * 4)); EPA_CUDA_CHECK(cudaMallocManaged(&per_owner, 12));
        EPA_CUDA_CHECK(cudaMallocManaged(&bad, 4));
        per_owner[0] = per_owner[1] = per_owner[2] = 0, *bad = 0;
        take_kernel<<<1, 1>>>(g, want, codes, per_owner, bad);
        EPA_CUDA_CHECK(cudaDeviceSynchronize());
        epa::ArenaMeta m0, m1, mh;
        EPA_CUDA_CHECK(cudaMemcpy(&m0, base[0], sizeof m0, cudaMemcpyDeviceToHost));
        EPA_CUDA_CHECK(cudaSetDevice(1));
        EPA_CUDA_CHECK(cudaMemcpy(&m1, base[1], sizeof m1, cudaMemcpyDeviceToHost));
        EPA_CUDA_CHECK(cudaSetDevice(0));
        EPA_CUDA_CHECK(cudaMemcpy(&mh, host.ring, sizeof mh, cudaMemcpyDeviceToHost));
        char buf[256];
        std::snprintf(buf, sizeof buf, "fallback: %d local, %d from GPU 1, %d from host (want 8, 8, 24); %d bad pages; all returned: %d/%d/%d free",
                      per_owner[0], per_owner[1], per_owner[2], *bad, m0.free_pages, m1.free_pages, mh.free_pages);
        expect(per_owner[0] == 8 and per_owner[1] == 8 and per_owner[2] == 24 and *bad == 0 and
               m0.free_pages == n_gpu and m1.free_pages == n_gpu and mh.free_pages == n_host, buf);
        host.destroy();
    }
    {   // 2. PagedWeights
        EPA_CUDA_CHECK(cudaSetDevice(0));
        EPA_CU_CHECK(cuInit(0));
        const int N = 4;
        const int64_t lb = 64ll << 20, tb = 8ll << 20;
        char* src;
        EPA_CUDA_CHECK(cudaMalloc(&src, lb));
        epa::PagedWeights w;
        w.create(0, N, lb, tb);
        for (int k = 0; k < N; ++k) {
            fill_kernel<<<256, 256>>>(src, lb, k);
            w.load(k, src, 0);
            EPA_CUDA_CHECK(cudaStreamSynchronize(0));
        }
        unsigned long long* bad;
        EPA_CUDA_CHECK(cudaMallocManaged(&bad, 8));
        *bad = 0;
        cudaStream_t s;
        EPA_CUDA_CHECK(cudaStreamCreate(&s));
        for (int round = 0; round < 2; ++round)
            for (int k = 0; k < N; ++k) {
                w.page_in(k, s, 2 << 20);
                EPA_CUDA_CHECK(cudaStreamSynchronize(s));
                check_kernel<<<256, 256>>>(static_cast<const char*>(w.layer(k)), lb, k, bad);
                EPA_CUDA_CHECK(cudaDeviceSynchronize());
            }
        char buf[256];
        std::snprintf(buf, sizeof buf, "PagedWeights: %d layers x %lld MB, tail %lld MB shared: %llu bad bytes over 8 page-ins; HBM %lld MB, saved %lld MB",
                      N, (long long)(lb >> 20), (long long)(w.tail_bytes >> 20), *bad, (long long)(w.hbm_bytes() >> 20), (long long)(w.saved_bytes() >> 20));
        expect(*bad == 0 and w.saved_bytes() == 3 * tb and w.hbm_bytes() == N * (lb - tb) + tb, buf);
        w.destroy();
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "all host tier tests passed\n", g_fail);
    return g_fail != 0;
}
