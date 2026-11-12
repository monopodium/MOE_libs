// Host side of EPArena: one plain cudaMalloc per rank, shared with peers through
// cudaIpc handles (no VMM, no NVSHMEM), and a lifetime-aware region planner.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace epa {

#define EPA_CUDA_CHECK(cmd)                                                            \
    do {                                                                               \
        cudaError_t e_ = (cmd);                                                        \
        if (e_ != cudaSuccess)                                                         \
            throw std::runtime_error(std::string("EPA CUDA error: ") + cudaGetErrorString(e_) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

// ---------------------------------------------------------------------------------
// Region planner
// ---------------------------------------------------------------------------------
// A region is live during a set of phases (bitmask). Two regions may share bytes iff
// their phase masks are disjoint. Planning is first-fit by decreasing size over the
// interference graph, which is optimal for the handful of regions EP needs and is
// the same heuristic XLA/TVM buffer assignment uses.
struct RegionSpec {
    std::string name;
    int64_t bytes;
    uint32_t phases;   // bit i set: live in phase i
    int64_t align = 256;
};

struct PlannedRegion {
    std::string name;
    int64_t offset, bytes;
    uint32_t phases;
};

inline std::vector<PlannedRegion> plan_regions(std::vector<RegionSpec> specs, int64_t* total_bytes) {
    std::vector<int> order(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return specs[a].bytes > specs[b].bytes; });

    std::vector<PlannedRegion> placed;
    int64_t total = 0;
    for (int idx : order) {
        const auto& s = specs[idx];
        // Collect intervals of already-placed regions that interfere with `s`
        std::vector<std::pair<int64_t, int64_t>> busy;
        for (const auto& p : placed)
            if (p.phases & s.phases) busy.emplace_back(p.offset, p.offset + p.bytes);
        std::sort(busy.begin(), busy.end());
        int64_t off = 0;
        for (const auto& [b, e] : busy) {
            off = (off + s.align - 1) / s.align * s.align;
            if (off + s.bytes <= b) break;
            off = std::max(off, e);
        }
        off = (off + s.align - 1) / s.align * s.align;
        placed.push_back({s.name, off, s.bytes, s.phases});
        total = std::max(total, off + s.bytes);
    }
    // Return in spec order
    std::vector<PlannedRegion> out;
    for (const auto& s : specs)
        for (const auto& p : placed)
            if (p.name == s.name) out.push_back(p);
    if (total_bytes) *total_bytes = total;
    return out;
}

// ---------------------------------------------------------------------------------
// Symmetric heap
// ---------------------------------------------------------------------------------
// Usage (all ranks, same `bytes`):
//   SymmetricHeap h(bytes);                      // cudaMalloc
//   auto mine = h.ipc_handle();                  // exchange with an all-gather
//   h.open_peers(all_handles, rank);             // cudaIpcOpenMemHandle, device peer table
struct SymmetricHeap {
    void* base = nullptr;
    int64_t bytes = 0;
    int rank = -1, num_ranks = 0;
    std::vector<void*> peer_bases;     // host copy
    void** d_peer_bases = nullptr;     // device copy (for PeerTable)

    explicit SymmetricHeap(int64_t n) : bytes(n) {
        EPA_CUDA_CHECK(cudaMalloc(&base, std::max<int64_t>(n, 1)));
        EPA_CUDA_CHECK(cudaMemset(base, 0, std::max<int64_t>(n, 1)));
    }

    std::vector<uint8_t> ipc_handle() const {
        cudaIpcMemHandle_t h;
        EPA_CUDA_CHECK(cudaIpcGetMemHandle(&h, base));
        std::vector<uint8_t> v(sizeof(h));
        std::memcpy(v.data(), &h, sizeof(h));
        return v;
    }

    void open_peers(const std::vector<std::vector<uint8_t>>& handles, int my_rank) {
        rank = my_rank, num_ranks = static_cast<int>(handles.size());
        peer_bases.assign(num_ranks, nullptr);
        for (int r = 0; r < num_ranks; ++r) {
            if (r == rank) {
                peer_bases[r] = base;
                continue;
            }
            cudaIpcMemHandle_t h;
            std::memcpy(&h, handles[r].data(), sizeof(h));
            EPA_CUDA_CHECK(cudaIpcOpenMemHandle(&peer_bases[r], h, cudaIpcMemLazyEnablePeerAccess));
        }
        EPA_CUDA_CHECK(cudaMalloc(&d_peer_bases, sizeof(void*) * num_ranks));
        EPA_CUDA_CHECK(cudaMemcpy(d_peer_bases, peer_bases.data(), sizeof(void*) * num_ranks, cudaMemcpyHostToDevice));
    }

    ~SymmetricHeap() {
        for (int r = 0; r < num_ranks; ++r)
            if (r != rank and peer_bases[r] != nullptr) cudaIpcCloseMemHandle(peer_bases[r]);
        if (d_peer_bases) cudaFree(d_peer_bases);
        if (base) cudaFree(base);
    }
};

}  // namespace epa
