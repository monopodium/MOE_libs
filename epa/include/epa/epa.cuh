// EPArena (EPA): device-side, VMM-free memory placement for expert-parallel buffers.
//
// Everything a library uses is the Home + Window API in window.cuh:
//   epa::PeerTable  peer addressing on any symmetric-memory provider
//   epa::Home       producer-owned exact rows, read in place by peers
//   epa::Window     the bounded consumer staging and the round rule
// plus `Stats` below. Nothing here allocates: the host planner (heap.hpp) carves
// regions out of one cudaMalloc per rank.
#pragma once

#include <cuda_runtime.h>

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
