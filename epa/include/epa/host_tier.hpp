// epa host tier: pinned host memory as the last fallback (host side; include from .cu/.cpp).
//
// The fallback order of EPArena is: local pages -> lease IDLE pages -> borrow a peer's pages
// (Board) -> host memory. The host tier is used only when every GPU of the group is out of
// reserved memory; host memory is ~10x slower than NVLink (PCIe, ~55 GB/s measured on B300),
// so what lives there must be paged back in ahead of use.
//
//   HostArena     an epa::Arena whose ring state lives in device memory and whose pages live in
//                 pinned, device-mapped host memory (UVA): data_off = host pages - ring. Device
//                 code takes and gives its pages like any other arena (ArenaGroup::host).
//
//   PagedWeights  read-only data (expert weights) that a kernel needs as one contiguous tensor
//                 per layer (DeepGEMM's B), with only a slice resident at a time. N layers of
//                 layer_bytes each get their own contiguous virtual range [body_k | tail]; the
//                 body is the layer's own memory, and the tail of EVERY layer maps (CUDA VMM) to
//                 one shared physical region. All layers' tails wait in pinned host memory;
//                 page_in(k) copies layer k's tail into the shared region in pages, on a stream,
//                 so layer k's view is complete. The data is clean, so evicting = overwriting:
//                 nothing is written back. HBM saved: (N - 1) x tail_bytes.
//                 Rule for the caller: page_in(k) only after every reader of the previous layer's
//                 tail has finished, and every reader of layer k waits for it (events).
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <epa/arena.cuh>

namespace epa {

#define EPA_CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "epa: CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)
#define EPA_CU_CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(r_, &s_); std::fprintf(stderr, "epa: CU %s at %s:%d\n", s_ ? s_ : "?", __FILE__, __LINE__); std::abort(); } } while (0)

// ------------------------------------------------------------------ host arena
struct HostArena {
    Arena arena;              // ring in device memory, pages in pinned host memory
    void* ring = nullptr;     // device
    void* pages = nullptr;    // pinned host (device-mapped)

    static HostArena create(int num_pages, int64_t page_bytes) {
        HostArena h;
        EPA_CUDA_CHECK(cudaMalloc(&h.ring, Arena::meta_bytes(num_pages)));
        EPA_CUDA_CHECK(cudaHostAlloc(&h.pages, static_cast<size_t>(num_pages) * page_bytes,
                                     cudaHostAllocMapped | cudaHostAllocPortable));
        void* pages_dev = nullptr;
        EPA_CUDA_CHECK(cudaHostGetDevicePointer(&pages_dev, h.pages, 0));
        const auto img = Arena::host_init_image(num_pages);
        EPA_CUDA_CHECK(cudaMemcpy(h.ring, img.data(), img.size(), cudaMemcpyHostToDevice));
        h.arena.base = static_cast<char*>(h.ring);
        h.arena.page_bytes = page_bytes;
        h.arena.num_pages = num_pages;
        h.arena.data_off = static_cast<char*>(pages_dev) - static_cast<char*>(h.ring);
        return h;
    }
    void destroy() {
        if (ring) EPA_CUDA_CHECK(cudaFree(ring));
        if (pages) EPA_CUDA_CHECK(cudaFreeHost(pages));
        ring = pages = nullptr;
    }
};

// ------------------------------------------------------------------ paged read-only weights
struct PagedWeights {
    int device = 0, num_layers = 0;
    int64_t layer_bytes = 0, tail_bytes = 0, body_bytes = 0;
    CUdeviceptr va = 0;
    std::vector<CUmemGenericAllocationHandle> bodies;
    CUmemGenericAllocationHandle tail = 0;
    std::vector<char*> host_tails;   // pinned
    int resident = -1;               // whose tail the shared region holds (-1: none yet)

    static int64_t granularity(int device) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        size_t g = 0;
        EPA_CU_CHECK(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        return static_cast<int64_t>(g);
    }
    // tail_bytes is rounded down to the granularity (2 MB on B300); layer_bytes must be a multiple of it
    void create(int dev, int layers, int64_t layer_b, int64_t tail_b) {
        device = dev, num_layers = layers, layer_bytes = layer_b;
        const int64_t g = granularity(dev);
        if (layer_b % g != 0) { std::fprintf(stderr, "epa: PagedWeights layer_bytes %% %lld != 0\n", (long long) g); std::abort(); }
        tail_bytes = tail_b / g * g;
        body_bytes = layer_bytes - tail_bytes;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        EPA_CU_CHECK(cuMemAddressReserve(&va, static_cast<size_t>(layers) * layer_bytes, g, 0, 0));
        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE, acc.location.id = dev;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (tail_bytes > 0) EPA_CU_CHECK(cuMemCreate(&tail, tail_bytes, &prop, 0));
        bodies.assign(layers, 0);
        host_tails.assign(layers, nullptr);
        for (int k = 0; k < layers; ++k) {
            const CUdeviceptr at = va + static_cast<CUdeviceptr>(k) * layer_bytes;
            if (body_bytes > 0) {
                EPA_CU_CHECK(cuMemCreate(&bodies[k], body_bytes, &prop, 0));
                EPA_CU_CHECK(cuMemMap(at, body_bytes, 0, bodies[k], 0));
            }
            if (tail_bytes > 0) {
                EPA_CU_CHECK(cuMemMap(at + body_bytes, tail_bytes, 0, tail, 0));   // the same physical tail
                EPA_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_tails[k]), tail_bytes, cudaHostAllocPortable));
            }
            EPA_CU_CHECK(cuMemSetAccess(at, layer_bytes, &acc, 1));
        }
    }
    void* layer(int k) const { return reinterpret_cast<void*>(va + static_cast<CUdeviceptr>(k) * layer_bytes); }
    int64_t saved_bytes() const { return static_cast<int64_t>(num_layers - 1) * tail_bytes; }
    int64_t hbm_bytes() const { return static_cast<int64_t>(num_layers) * body_bytes + tail_bytes; }

    // Setup: layer k's full contents from device memory `src` (body to HBM, tail to host)
    void load(int k, const void* src, cudaStream_t stream) {
        EPA_CUDA_CHECK(cudaMemcpyAsync(layer(k), src, body_bytes, cudaMemcpyDeviceToDevice, stream));
        if (tail_bytes > 0)
            EPA_CUDA_CHECK(cudaMemcpyAsync(host_tails[k], static_cast<const char*>(src) + body_bytes, tail_bytes,
                                           cudaMemcpyDeviceToHost, stream));
    }
    // Page layer k's tail into the shared region, page_bytes per copy (copy engine, no SMs)
    void page_in(int k, cudaStream_t stream, int64_t page_bytes) {
        char* dst = static_cast<char*>(layer(k)) + body_bytes;
        for (int64_t off = 0; off < tail_bytes; off += page_bytes) {
            const int64_t n = off + page_bytes <= tail_bytes ? page_bytes : tail_bytes - off;
            EPA_CUDA_CHECK(cudaMemcpyAsync(dst + off, host_tails[k] + off, n, cudaMemcpyHostToDevice, stream));
        }
        resident = k;
    }
    void destroy() {
        if (!va) return;
        for (int k = 0; k < num_layers; ++k) {
            const CUdeviceptr at = va + static_cast<CUdeviceptr>(k) * layer_bytes;
            if (body_bytes > 0) { EPA_CU_CHECK(cuMemUnmap(at, body_bytes)); EPA_CU_CHECK(cuMemRelease(bodies[k])); }
            if (tail_bytes > 0) { EPA_CU_CHECK(cuMemUnmap(at + body_bytes, tail_bytes)); EPA_CUDA_CHECK(cudaFreeHost(host_tails[k])); }
        }
        if (tail_bytes > 0) EPA_CU_CHECK(cuMemRelease(tail));
        EPA_CU_CHECK(cuMemAddressFree(va, static_cast<size_t>(num_layers) * layer_bytes));
        va = 0;
    }
};

}  // namespace epa
