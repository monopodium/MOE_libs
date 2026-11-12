// Microbenchmarks for EPArena's primitives on one NVLink node:
//   - push (st to peer) vs pull (ld from peer) of fixed-size rows
//   - NVLink fetch-add latency and warp-aggregated reservation throughput
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <epa/epa.cuh>
#include <epa/heap.hpp>

#include <memory>

namespace {

std::unique_ptr<epa::SymmetricHeap> g_heap;

int64_t heap_create(int64_t bytes) {
    g_heap = std::make_unique<epa::SymmetricHeap>(bytes);
    return reinterpret_cast<int64_t>(g_heap->base);
}

torch::Tensor heap_handle() {
    auto v = g_heap->ipc_handle();
    auto t = torch::empty({static_cast<int64_t>(v.size())}, torch::kUInt8);
    std::memcpy(t.data_ptr(), v.data(), v.size());
    return t;
}

void heap_open(torch::Tensor all, int64_t rank) {   // [R, 64] uint8 (CPU)
    std::vector<std::vector<uint8_t>> hs(all.size(0));
    for (int64_t r = 0; r < all.size(0); ++r)
        hs[r].assign(all[r].data_ptr<uint8_t>(), all[r].data_ptr<uint8_t>() + all.size(1));
    g_heap->open_peers(hs, static_cast<int>(rank));
}

void heap_free() { g_heap.reset(); }

epa::PeerTable table() { return {g_heap->base, g_heap->d_peer_bases}; }

// Each block handles rows [b * rows_per_block, ...). Row i goes to/from peer (i % R).
// push: local src rows -> peer dst rows; pull: peer src rows -> local dst rows.
template <bool kPull>
__global__ void copy_rows_kernel(epa::PeerTable pt, int64_t src_off, int64_t dst_off, int num_rows,
                                 int row_int4, int num_ranks, int rank) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nwarps = blockDim.x / 32;
    for (int i = blockIdx.x * nwarps + warp; i < num_rows; i += gridDim.x * nwarps) {
        const int peer = (rank + 1 + i % (num_ranks - 1)) % num_ranks;   // never self
        const int4* src;
        int4* dst;
        const auto lsrc = reinterpret_cast<int4*>(static_cast<char*>(pt.local_base) + src_off) + static_cast<int64_t>(i) * row_int4;
        const auto ldst = reinterpret_cast<int4*>(static_cast<char*>(pt.local_base) + dst_off) + static_cast<int64_t>(i) * row_int4;
        if (kPull) src = pt.peer<int4>(lsrc, peer), dst = ldst;
        else src = lsrc, dst = pt.peer<int4>(ldst, peer);
        constexpr int U = 8;
        int4 v[U];
        int j = lane;
        for (; j + 32 * (U - 1) < row_int4; j += 32 * U) {
            #pragma unroll
            for (int u = 0; u < U; ++u) v[u] = __ldg(src + j + 32 * u);
            #pragma unroll
            for (int u = 0; u < U; ++u) dst[j + 32 * u] = v[u];
        }
        for (; j < row_int4; j += 32) dst[j] = __ldg(src + j);
    }
}

// Single-thread dependent fetch-adds to `peer`: measures NVLink atomic round trip.
__global__ void atomic_latency_kernel(epa::PeerTable pt, int64_t ctr_off, int peer, int iters,
                                      long long* out_cycles) {
    int* c = pt.peer<int>(static_cast<char*>(pt.local_base) + ctr_off, peer);
    int r = 0;
    const long long t0 = clock64();
    for (int i = 0; i < iters; ++i) r = atomicAdd(c + (r >> 30), 1);   // address depends on the last result
    const long long t1 = clock64();
    out_cycles[0] = t1 - t0;
    out_cycles[1] = r;
}

// Every warp reserves `n_per_lane` rows on `dst = lane % R` for `iters` rounds,
// either one fetch-add per lane or warp-aggregated.
template <bool kAggregate>
__global__ void reserve_kernel(epa::PeerTable pt, int64_t ctr_off, int num_ranks, int iters, int* sink) {
    int* head = reinterpret_cast<int*>(static_cast<char*>(pt.local_base) + ctr_off);
    epa::Window win;
    win.shared = 1 << 30, win.max_rows = 1 << 30;
    const int lane = threadIdx.x % 32;
    int64_t acc = 0;
    for (int it = 0; it < iters; ++it) {
        const int dst = (lane + it) % num_ranks;
        if (kAggregate) acc += win.warp_reserve(pt, head, dst, 1);
        else acc += win.accept(atomicAdd(pt.peer<int>(head, dst), 1), 1);
    }
    if (acc == 0x7fffffff) sink[0] = static_cast<int>(acc);
}

float time_it(const std::function<void(cudaStream_t)>& fn, int iters) {
    auto s = at::cuda::getCurrentCUDAStream().stream();
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    fn(s);
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; ++i) fn(s);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a), cudaEventDestroy(b);
    return ms / iters;
}

double copy_rows(bool pull, int64_t src_off, int64_t dst_off, int64_t num_rows, int64_t row_bytes,
                 int64_t num_blocks, int64_t threads, int64_t iters) {
    auto pt = table();
    const int R = g_heap->num_ranks, rank = g_heap->rank;
    return time_it([&](cudaStream_t s) {
        if (pull) copy_rows_kernel<true><<<num_blocks, threads, 0, s>>>(pt, src_off, dst_off, num_rows, row_bytes / 16, R, rank);
        else copy_rows_kernel<false><<<num_blocks, threads, 0, s>>>(pt, src_off, dst_off, num_rows, row_bytes / 16, R, rank);
    }, iters);
}

double atomic_latency(int64_t ctr_off, int64_t peer, int64_t iters) {
    auto out = torch::zeros({2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto s = at::cuda::getCurrentCUDAStream().stream();
    atomic_latency_kernel<<<1, 1, 0, s>>>(table(), ctr_off, peer, iters, reinterpret_cast<long long*>(out.data_ptr()));
    int clk_khz;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    const auto cyc = out.cpu()[0].item<int64_t>();
    return static_cast<double>(cyc) / iters / (clk_khz * 1e-6);   // ns per fetch-add (SM clock)
}

double reserve_rate(bool aggregate, int64_t ctr_off, int64_t blocks, int64_t threads, int64_t iters) {
    auto sink = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto pt = table();
    const int R = g_heap->num_ranks;
    return time_it([&](cudaStream_t s) {
        if (aggregate) reserve_kernel<true><<<blocks, threads, 0, s>>>(pt, ctr_off, R, iters, sink.data_ptr<int>());
        else reserve_kernel<false><<<blocks, threads, 0, s>>>(pt, ctr_off, R, iters, sink.data_ptr<int>());
    }, 5);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("heap_create", &heap_create);
    m.def("heap_handle", &heap_handle);
    m.def("heap_open", &heap_open);
    m.def("heap_free", &heap_free);
    m.def("copy_rows", &copy_rows);
    m.def("atomic_latency", &atomic_latency);
    m.def("reserve_rate", &reserve_rate);
}
