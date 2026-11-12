#pragma once

// EPArena segment (expanded layout, no CPU sync): recv_x is one contiguous region of C rows
// at the tail of the EP buffer, and y is leased into the same rows by the GEMMs. Rows of this
// call beyond C go to ranges borrowed from the unused tail of other GPUs' segments: in
// dispatch's notify block every GPU publishes its own end (epoch-stamped), and a GPU whose
// rows exceed C claims [own_end, C) ranges of the lenders with one CAS each. The copy
// epilogue writes rows >= C there, and combine reads them back through the same table.
//
// Header (4096 bytes, before the rows), ints:
//   [0] C   [1] epoch   [2] own_end   [3] claim_next   [4] chunks   [5] rows this call
//   [6] rows not placed this call   [7] rows borrowed (total)   [8] S: rows of the local spill
//   scratch (the first S rows past C go there, no NVLink; borrowing starts after them)
//   [16 + 4j] chunk j: lender, first logical row, rows, lender's row      (j < 16)
//   bytes [512, 512 + 8R): this GPU's address of rank r's segment rows; [512 + 8R]: the local
//   spill scratch (chunk lender R)
// A lender's tail is free again at its next call: every borrower has finished that call's
// combine before it sends the next call's counts.
#include <cstdint>

namespace deep_ep::layout {

struct EPASeg {
    int* hd;
    static constexpr int kHeaderBytes = 4096, kMaxChunks = 16;

    __host__ __device__ int cap() const { return hd[0]; }
    __host__ __device__ unsigned long long* bases() const { return reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(hd) + 512); }
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
};

}  // namespace deep_ep::layout
