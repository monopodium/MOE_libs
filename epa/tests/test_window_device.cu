// Device-side checks of epa::Window / epa::Home on one GPU (peer table = self, so
// every "rank" aliases the local heap):
//   warp_reserve  every lane gets a distinct base; bases of one destination are dense
//                 and accepted only while they fit the shared rows
//   wait_free     a producer block fills 4 rounds of a 64-slot window while a consumer
//                 block frees slots; every row arrives exactly once, in order per slot
//   Home::row     addresses row i of rank r's region
// Build: nvcc -std=c++17 -arch=sm_100a -I ../include test_window_device.cu -o /tmp/twd
#include <cstdio>
#include <vector>

#include <epa/epa.cuh>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s\n", cudaGetErrorString(e_)); return 1; } } while (0)

constexpr int kRanks = 4;

__global__ void reserve_kernel(epa::PeerTable pt, int* heads, epa::Window win, long long* out) {
    const int lane = threadIdx.x & 31, warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int dst = (lane + warp) % kRanks, n = 1 + lane % 3;
    // a distinct head per destination: heads[dst] (all in the local heap)
    const auto base = win.warp_reserve(pt, heads + dst, dst, n);
    out[(warp * 32 + lane) * 3 + 0] = dst;
    out[(warp * 32 + lane) * 3 + 1] = n;
    out[(warp * 32 + lane) * 3 + 2] = base;
}

__global__ void ring_kernel(epa::Window win, int* slots, unsigned* freed, int* seen, int n_rows) {
    if (blockIdx.x == 0) {           // producer: fills row r into slot(r) once it is free
        for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
            win.wait_free(freed, r);
            reinterpret_cast<volatile int*>(slots)[win.slot(r)] = r + 1;
        }
    } else {                         // consumer: takes rows in order per slot, frees them
        for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
            int v;
            while ((v = reinterpret_cast<volatile int*>(slots)[win.slot(r)]) != r + 1) __nanosleep(32);
            atomicAdd(seen + r, 1);
            reinterpret_cast<volatile int*>(slots)[win.slot(r)] = 0;
            __threadfence();
            atomicAdd(freed + win.slot(r), 1u);
        }
    }
}

__global__ void home_kernel(epa::Home h, long long* out) {
    out[0] = reinterpret_cast<long long>(h.row<char>(2, 5)) - reinterpret_cast<long long>(h.local_row<char>(0));
}

int main() {
    // heap with a self-only peer table for kRanks "ranks"
    char* heap;
    CHECK(cudaMalloc(&heap, 1 << 20));
    CHECK(cudaMemset(heap, 0, 1 << 20));
    std::vector<void*> bases(kRanks, heap);
    void** d_bases;
    CHECK(cudaMalloc(&d_bases, sizeof(void*) * kRanks));
    CHECK(cudaMemcpy(d_bases, bases.data(), sizeof(void*) * kRanks, cudaMemcpyHostToDevice));
    epa::PeerTable pt{heap, d_bases};
    int fails = 0;

    // 1. warp_reserve
    {
        epa::Window win;
        win.shared = 150, win.max_rows = 1 << 20;
        const int warps = 16, n_lanes = warps * 32;
        long long* d_out;
        CHECK(cudaMalloc(&d_out, n_lanes * 3 * sizeof(long long)));
        reserve_kernel<<<4, 128>>>(pt, reinterpret_cast<int*>(heap), win, d_out);
        CHECK(cudaDeviceSynchronize());
        std::vector<long long> o(n_lanes * 3);
        CHECK(cudaMemcpy(o.data(), d_out, o.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<int> heads(kRanks);
        CHECK(cudaMemcpy(heads.data(), heap, kRanks * 4, cudaMemcpyDeviceToHost));
        for (int d = 0; d < kRanks; ++d) {
            std::vector<int> used(heads[d], 0);
            long long total = 0;
            for (int i = 0; i < n_lanes; ++i) {
                if (o[i * 3] != d) continue;
                total += o[i * 3 + 1];
                const long long b = o[i * 3 + 2];
                if (b >= 0) {
                    if (b + o[i * 3 + 1] > win.shared) ++fails, printf("reserve: accepted past window\n");
                    for (long long j = b; j < b + o[i * 3 + 1]; ++j) used[j]++;
                }
            }
            if (total != heads[d]) ++fails, printf("reserve: head %d != requested %lld\n", heads[d], total);
            for (int j = 0; j < std::min<int>(heads[d], win.shared); ++j)
                if (used[j] > 1) ++fails, printf("reserve: row %d of dst %d given twice\n", j, d);
        }
        printf("warp_reserve: %s (heads %d %d %d %d, window %lld)\n", fails ? "FAIL" : "ok",
               heads[0], heads[1], heads[2], heads[3], (long long)win.shared);
    }

    // 2. wait_free: 4 rounds through a 64-slot window
    {
        const int W = 64, n_rows = 4 * W;
        epa::Window win;
        win.shared = W, win.max_rows = n_rows;
        int *slots, *seen;
        unsigned* freed;
        CHECK(cudaMalloc(&slots, W * 4)); CHECK(cudaMemset(slots, 0, W * 4));
        CHECK(cudaMalloc(&freed, W * 4)); CHECK(cudaMemset(freed, 0, W * 4));
        CHECK(cudaMalloc(&seen, n_rows * 4)); CHECK(cudaMemset(seen, 0, n_rows * 4));
        ring_kernel<<<2, W>>>(win, slots, freed, seen, n_rows);
        CHECK(cudaDeviceSynchronize());
        std::vector<int> s(n_rows);
        CHECK(cudaMemcpy(s.data(), seen, n_rows * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int v : s) bad += v != 1;
        fails += bad;
        printf("wait_free: %s (%d rows through %d slots, %lld rounds)\n", bad ? "FAIL" : "ok", n_rows, W,
               (long long)win.num_rounds());
    }

    // 3. Home::row
    {
        epa::Home h{pt, heap + 4096, 100};
        long long* d_out;
        CHECK(cudaMalloc(&d_out, 8));
        home_kernel<<<1, 1>>>(h, d_out);
        long long off;
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&off, d_out, 8, cudaMemcpyDeviceToHost));
        fails += off != 500;
        printf("Home::row: %s\n", off == 500 ? "ok" : "FAIL");
    }
    printf(fails ? "FAILED\n" : "all device checks passed\n");
    return fails != 0;
}
