// EPArena segment: one GPU's receive rows as one contiguous region of C rows, plus rows
// borrowed from the unused tails of other GPUs' segments when a call needs more than C.
//
// The dispatch that fills a segment writes this header (in DeepEP: the notify block of
// dispatch, expanded layout, no CPU sync); every consumer reads rows through `row()`, so
// none of them needs to know where a row was placed.
//
// Header (kHeaderBytes, right before the C rows), ints:
//   [0] C   [1] epoch (+1 per call)   [2] own_end   [3] claim_next   [4] chunks
//   [5] rows this call   [6] rows not placed this call   [7] rows borrowed (total)
//   [16 + 4j] chunk j: lender, first logical row, rows, lender's row      (j < kMaxChunks)
//   bytes [512, 512 + 8R): this GPU's address of rank r's segment rows; [512 + 8R]: a local
//   buffer the caller may point chunks at (lender R), e.g. where the borrowed rows' y is leased
// A lender's tail is free again at its next call: every borrower has finished that call's
// combine before it sends the next call's counts.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EPA_SEG_HD __host__ __device__ __forceinline__
#else
#define EPA_SEG_HD inline
#endif

namespace epa {

struct Segment {
    int* hd;
    static constexpr int kHeaderBytes = 4096, kMaxChunks = 16;

    EPA_SEG_HD int cap() const { return hd[0]; }
    EPA_SEG_HD unsigned long long* bases() const { return reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(hd) + 512); }
    // This GPU's own C rows (they follow the header)
    EPA_SEG_HD void* local_rows() const { return reinterpret_cast<char*>(hd) + kHeaderBytes; }

#if defined(__CUDACC__)
    // Address of logical row `idx` (local rows: `local`), or nullptr if it was not placed
    __device__ __forceinline__ void* row(const int& idx, void* local, const int64_t& row_bytes) const {
        if (idx < hd[0]) return static_cast<char*>(local) + static_cast<int64_t>(idx) * row_bytes;
        const int n = hd[4];
        for (int j = 0; j < n; ++ j) {
            const int* c = hd + 16 + 4 * j;
            if (idx < c[1] + c[2])
                return reinterpret_cast<char*>(bases()[c[0]]) + static_cast<int64_t>(c[3] + idx - c[1]) * row_bytes;
        }
        return nullptr;
    }
#endif
};

}  // namespace epa
