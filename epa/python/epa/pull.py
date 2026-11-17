"""epa.pull: pull rows again from where they wait (their Home), instead of keeping a copy.

A receive list whose rows are only needed at the start and the end of a call need not
hold them in between: the rows still sit in their producers' buffers (the Home of
window.cuh), so the consumer drops its copy, lends the space to other tensors, and pulls
the rows back just before the last use. Over NVLink this costs one read of the rows,
far less than splitting the call into rounds.

    pull_rows(dst, dst_offset_1, src_ptrs, rows_1, row_bytes, run_1=lean_1)

`src_ptrs[i]` is the device address of row i (local or a peer's, e.g. a symmetric
buffer); row i lands at dst + dst_offset + i * row_bytes. Rows, offset and the run flag
are device tensors, so the launch is CUDA-graph safe and never syncs: when run_1 is 0
the kernel exits at once.
"""
from __future__ import annotations

import torch

__all__ = ['pull_rows']


def pull_rows(dst: torch.Tensor, dst_offset_1: torch.Tensor, src_ptrs: torch.Tensor, rows_1: torch.Tensor,
              row_bytes: int, run_1: torch.Tensor | None = None, num_programs: int | None = None) -> None:
    """dst: byte buffer (uint8); dst_offset_1: device [1] int64 byte offset; src_ptrs: device
    int64 row addresses; rows_1: device [1] rows to pull; run_1: optional device [1] flag."""
    if row_bytes % 16:
        raise ValueError('row_bytes must be a multiple of 16')
    if num_programs is None:
        num_programs = torch.cuda.get_device_properties(dst.device).multi_processor_count * 2
    words = row_bytes // 8
    block = min(4096, 1 << (words - 1).bit_length())
    _pull_kernel[(num_programs,)](dst, dst_offset_1, src_ptrs, rows_1, run_1 if run_1 is not None else rows_1,
                                  WORDS=words, BLOCK=block, HAS_RUN=run_1 is not None)


def _make_kernel():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def kernel(dst_ptr, dst_off_ptr, src_ptrs, rows_ptr, run_ptr,
               WORDS: tl.constexpr, BLOCK: tl.constexpr, HAS_RUN: tl.constexpr):
        run = 1
        if HAS_RUN:
            run = tl.load(run_ptr)
        if run != 0:
            rows = tl.load(rows_ptr).to(tl.int64)
            dst = (dst_ptr + tl.load(dst_off_ptr)).to(tl.pointer_type(tl.int64))
            k = tl.arange(0, BLOCK)
            for r in tl.range(tl.program_id(0), rows, tl.num_programs(0)):
                src = tl.load(src_ptrs + r).to(tl.pointer_type(tl.int64), bitcast=True)
                for w0 in tl.static_range(0, WORDS, BLOCK):
                    m = w0 + k < WORDS
                    tl.store(dst + r * WORDS + w0 + k, tl.load(src + w0 + k, mask=m), mask=m)

    return kernel


_pull_kernel = _make_kernel()
