// EPArena (EPA): device-side memory placement for expert-parallel libraries.
//
// No size is chosen for the worst case. A consumer gets a bounded buffer; rows that do
// not fit wait where they already are (their producer's Home, or a later round), and every
// decision is made on the device, so nothing syncs with the host and CUDA graphs stay fixed.
//
// Headers, by what a library needs (each one documents its own interface):
//
//   rules (header only, host + device)
//     window.cuh     Home + Window: row r runs in round r / W at slot r % W; PeerTable
//     pool.cuh       Pool: kept state at the bottom (LIFO), a per-call frame at the top
//                    that takes what is free; a reserve guarantees the frame a minimum W
//   device allocators
//     arena.cuh      Arena page ring (take / give from kernels, local or peer), ArenaGroup
//                    (fallback to peers' arenas), WarpHeap (malloc / free), PagedArray
//     space.cuh      Space: cross-GPU paged objects (page table, borrow cache, leases)
//   pieces for a receive buffer (DeepEP)
//     segment.cuh    Segment: a GPU's receive rows plus rows borrowed from peers' tails
//     row_pull.cuh   RowPull: a GEMM's A rows staged from a Segment, local or peer
//   host side
//     heap.hpp       one cudaMalloc per rank shared through cudaIpc; lifetime planner
//     host_tier.hpp  pinned host memory as the last tier; PagedWeights
//
// Python (python/epa/): epa.window, epa.rounds, epa.pool mirror the rules; epa.arena and
// epa.host wrap the allocators for PyTorch callers.
//
// This header pulls in the rules and `Stats`; include the others where they are used.
#pragma once

#include <cuda_runtime.h>

#include <epa/pool.cuh>
#include <epa/window.cuh>

namespace epa {

// Device-side counters the host reads every N steps (no sync on the step) to re-size
// the window: high-water mark per key and rows that were read from Home.
struct Stats {
    int* max_per_key;     // [num_keys]
    int* num_spilled;     // [1]

    __device__ __forceinline__ void record(int key, int count, int spilled) const {
        if (max_per_key != nullptr) atomicMax(max_per_key + key, count);
        if (num_spilled != nullptr and spilled > 0) atomicAdd(num_spilled, spilled);
    }
};

}  // namespace epa
