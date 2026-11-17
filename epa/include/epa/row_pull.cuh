// EPArena row pull: a GEMM whose A rows live in an EPArena segment, some of them on other GPUs.
//
// The GEMM keeps loading A from one local buffer (`stage`) with its own TMA map. Row j of the
// stage is logical row base_row + j of the segment; RowPull fills it from wherever the segment
// placed that row (this GPU or a lender), once, before any block loads it.
//
// Why not load straight from the lender: peer memory is not cached in this GPU's L2, and a GEMM
// reads every A row once per N-block, so each row would cross NVLink N / BLOCK_N times. The pull
// crosses it once per row.
//
// The stage is filled in groups of kGroupRows rows; each group is stamped when it lands, and a
// group already stamped is never copied again. Two ways to fill it, mixable:
//
//   in the consuming GEMM (spare warps, the blocks that read a tile share its rows):
//     const auto tok = pull.token();                                  once, after the grid dependency wait
//     pull.pull_rows(r0, r1, part, num_parts, lane, tok);             each spare warp, per block it is given
//     pull.wait_rows(r0, r1, tok);                                    the A-loading thread, before the tile's loads
//
//   ahead of time, in any earlier kernel on the same stream (e.g. a GEMM still busy with local
//   rows), so the consumer finds every group stamped and only checks:
//     pull.prefetch(worker, num_workers, lane, pull.token());         every spare warp of that kernel
//
// Nothing to release: a stage is refilled only after its consumer finished (stream order).
// Stamps hold (segment epoch, base_row), so they need no reset: a stamp from an earlier call or
// from another piece using the same stage never matches.
#pragma once

#include <cstdint>

#include <epa/segment.cuh>

namespace epa {

struct RowPull {
    int* hd;                          // segment header (epa/segment.cuh); nullptr: disabled
    void* stage;                      // local rows: stage row j <- logical row base_row + j
    unsigned long long* stamps;       // [stamp_count(stage_rows)], zero-initialized once
    int base_row;                     // logical row of stage row 0
    int row_bytes;                    // multiple of 16
    int stage_rows;                   // rows the stage holds

    static constexpr int kGroupRows = 2;

    // Host helpers
    static RowPull disabled() { return RowPull{nullptr, nullptr, nullptr, 0, 0, 0}; }
    EPA_SEG_HD static int64_t stamp_count(int64_t stage_rows) { return (stage_rows + kGroupRows - 1) / kGroupRows; }
    EPA_SEG_HD bool enabled() const { return hd != nullptr; }

#if defined(__CUDACC__)
    __device__ __forceinline__ unsigned long long token() const {
        const int epoch = *reinterpret_cast<volatile int*>(hd + 1);
        return (static_cast<unsigned long long>(static_cast<unsigned>(epoch)) << 32) | static_cast<unsigned>(base_row);
    }

    __device__ __forceinline__ bool stamped(const int& g, const unsigned long long& tok) const {
        return *reinterpret_cast<volatile unsigned long long*>(stamps + g) == tok;
    }

    // Stage rows that hold a row of this call (the rest are never copied, stamped or waited for)
    __device__ __forceinline__ int valid_rows() const {
        return max(0, min(stage_rows, hd[5] - base_row));
    }

    // One warp: copy the rows of group g into the stage (no fence, no stamp)
    __device__ __forceinline__ void copy_group_rows(const int& g, const int& lane) const {
        const Segment seg{hd};
        const int total = hd[5];
        const int j0 = g * kGroupRows, j1 = min(j0 + kGroupRows, stage_rows);
        const int vecs = row_bytes / 16;
        constexpr int kUnroll = 16;
        for (int j = j0; j < j1; ++ j) {
            if (base_row + j >= total) break;
            const auto src = static_cast<const int4*>(seg.row(base_row + j, seg.local_rows(), row_bytes));
            if (src == nullptr) continue;
            auto dst = reinterpret_cast<int4*>(static_cast<char*>(stage) + static_cast<int64_t>(j) * row_bytes);
            for (int v = lane; v < vecs; v += 32 * kUnroll) {
                int4 buf[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++ u)
                    if (v + u * 32 < vecs) buf[u] = __ldg(src + v + u * 32);
                #pragma unroll
                for (int u = 0; u < kUnroll; ++ u)
                    if (v + u * 32 < vecs) dst[v + u * 32] = buf[u];
            }
        }
    }

    // One warp: groups g0, g0 + stride, ... < g1 that are not in the stage yet are copied, then
    // published together: one fence (generic -> async proxy, GPU scope), then their stamps
    __device__ __forceinline__ void copy_groups(const int& g0, const int& g1, const int& stride,
                                                const int& lane, const unsigned long long& tok) const {
        bool any = false;
        for (int g = g0; g < g1; g += stride) {
            if (__shfl_sync(0xffffffff, lane == 0 ? stamped(g, tok) : false, 0)) continue;
            copy_group_rows(g, lane);
            any = true;
        }
        if (not any) return;
        __threadfence();
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __syncwarp();
        for (int g = g0 + lane * stride; g < g1; g += 32 * stride)   // same token: re-stamping is harmless
            *reinterpret_cast<volatile unsigned long long*>(stamps + g) = tok;
    }

    // Consumer side: stage rows [r0, r1) are read by `num_parts` (block, warp) visitors; this is one
    __device__ __forceinline__ void pull_rows(const int& r0, const int& r1, const int& part, const int& num_parts,
                                              const int& lane, const unsigned long long& tok) const {
        const int g0 = r0 / kGroupRows, g1 = (min(r1, valid_rows()) + kGroupRows - 1) / kGroupRows;
        copy_groups(g0 + part, g1, num_parts, lane, tok);
    }

    // Wait until stage rows [r0, r1) are in. All stamps are read with independent loads (one
    // round trip per poll), then one acquire fence. Call it from a warp that runs ahead of the
    // A-loading thread (a waiter warp, never a copier) and hand each tile over with
    // signal_ready / wait_ready: the A-loading thread then only reads shared memory, and its
    // TMA pipeline keeps running ahead across blocks.
    __device__ __forceinline__ void wait_rows(const int& r0, const int& r1, const unsigned long long& tok) const {
        const int g0 = r0 / kGroupRows, g1 = (min(r1, valid_rows()) + kGroupRows - 1) / kGroupRows;
        while (true) {
            bool ready = true;
            int g = g0;
            if (g < g1 and (g & 1)) ready &= stamped(g ++, tok);
            for (; g + 1 < g1; g += 2) {
                unsigned long long a, b;
                asm volatile("ld.volatile.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(stamps + g));
                ready &= (a == tok) & (b == tok);
            }
            if (g < g1) ready &= stamped(g, tok);
            if (ready) break;
            __nanosleep(64);
        }
        asm volatile("fence.acq_rel.gpu;" ::: "memory");   // acquire: the stamps' writers fenced before stamping
    }

    // Hand-off inside one CTA: the waiter warp has confirmed `n` tiles, in the order the
    // A-loading thread loads them (both walk the same block schedule)
    __device__ __forceinline__ static void signal_ready(int* smem_count, const int& n) {
        asm volatile("st.release.cta.shared.u32 [%0], %1;" :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_count))), "r"(n) : "memory");
    }
    __device__ __forceinline__ static void wait_ready(int* smem_count, const int& n) {
        const auto addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_count));
        int v;
        do {
            asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
        } while (v < n);
        asm volatile("fence.proxy.async.global;" ::: "memory");   // the rows were written by the generic proxy
    }

    // Ahead of time: the whole stage, spread over `num_workers` warps of any kernel
    __device__ __forceinline__ void prefetch(const int& worker, const int& num_workers,
                                             const int& lane, const unsigned long long& tok) const {
        copy_groups(worker, (valid_rows() + kGroupRows - 1) / kGroupRows, num_workers, lane, tok);
    }
#endif
};

}  // namespace epa
