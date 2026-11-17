// epa::Pool: one buffer shared by long-lived kept state and a per-call frame.
//
//   [ kept state, grows up, LIFO ............ | free | frame (this call), at the top ]
//   0                                        top                              bytes
//
// For stream-ordered libraries whose kernels need contiguous memory (grouped GEMMs):
//   keep(n)        push n bytes of state that outlives the call (e.g. activations saved
//                  for backward); fails, never blocks, when it would cut into `reserve`
//   pop(off)       free back to `off` (LIFO, e.g. the matching backward)
//   frame_rows(r)  rows of a frame of `row_bytes` per row that fit above the kept state,
//                  capped at r. The frame starts at the top and needs no release.
// A frame always holds at least min_window() rows, because kept state never enters the
// reserve: the caller runs ceil(rows / frame_rows) rounds (window.cuh, epa.rounds), so no
// size is chosen for the worst case. Everything is decided on the device from `top`.
//
// Python mirror: python/epa/pool.py (Triton helpers for libraries with Triton planners).
#pragma once

#include <cstdint>

#include <epa/window.cuh>

namespace epa {

struct Pool {
    char* base = nullptr;
    int64_t bytes = 0;                    // whole buffer
    int64_t reserve = 0;                  // top bytes kept state may never use
    unsigned long long* top = nullptr;    // device: end of kept state

    static constexpr int64_t kAlign = 128;   // every tensor starts on a 128-byte boundary
    static EPA_HD int64_t align(int64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

    // Host sizing: reserve that guarantees `min_window` rows to a frame of `num_tensors`
    // tensors whose row sizes sum to row_bytes, and the rows a reserve guarantees.
    static EPA_HD int64_t reserve_for(int64_t min_window, int64_t row_bytes, int num_tensors) {
        return min_window * row_bytes + num_tensors * kAlign;
    }
    EPA_HD int64_t min_window(int64_t row_bytes, int num_tensors) const {
        const int64_t r = (reserve - num_tensors * kAlign) / row_bytes;
        return r > 0 ? r : 0;
    }

#if defined(__CUDACC__)
    // One thread. Offset of n kept bytes, or -1 if they would enter the reserve.
    __device__ __forceinline__ int64_t keep(int64_t n) const {
        const int64_t off = static_cast<int64_t>(*top);
        const int64_t end = off + align(n);
        if (end > bytes - reserve) return -1;
        *top = static_cast<unsigned long long>(end);
        return off;
    }
    __device__ __forceinline__ void pop(int64_t off) const { *top = static_cast<unsigned long long>(off); }

    // Rows that fit in the frame, capped at `rows`.
    __device__ __forceinline__ int64_t frame_rows(int64_t row_bytes, int64_t rows, int num_tensors) const {
        int64_t fit = (bytes - static_cast<int64_t>(*top) - num_tensors * kAlign) / row_bytes;
        fit = fit < rows ? fit : rows;
        return fit > 0 ? fit : 0;
    }
    // Frame tensors are carved down from the end: the k-th call returns its offset.
    __device__ __forceinline__ static int64_t carve(int64_t* end, int64_t n) {
        *end -= align(n);
        return *end;
    }
#endif
};

}  // namespace epa
