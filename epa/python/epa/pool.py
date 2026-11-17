"""epa.pool: one buffer shared by long-lived kept state and a per-call frame (mirror of
include/epa/pool.cuh), for stream-ordered libraries whose kernels need contiguous memory.

    [ kept state, grows up, LIFO ............ | free | frame (this call), at the top ]
    0                                        top                              bytes

Kept state (activations saved for backward, a KV page run, ...) is pushed and popped from
the bottom. The frame is everything above the kept state: a call asks for `rows` rows of
`row_bytes` and gets min(rows, what fits) rows; it runs the rest in rounds (epa.rounds).
`reserve` bytes at the top are never given to kept state, so a frame always holds at least
reserve / row_bytes rows: that is the caller's min_window.

Everything is decided on the device (from the kept top the device holds), so it is CUDA-graph
safe and needs no host sync. Device helpers are Triton functions callable from a library's
own Triton kernels; host helpers size the buffer.
"""
from __future__ import annotations

__all__ = ['frame_rows', 'min_window', 'reserve_bytes', 'ALIGN']

ALIGN = 128   # every tensor in a frame starts on a 128-byte boundary


def reserve_bytes(min_window: int, row_bytes: int, num_tensors: int) -> int:
    """Bytes to reserve at the top so that a frame always holds `min_window` rows."""
    return min_window * row_bytes + num_tensors * ALIGN


def min_window(reserve: int, row_bytes: int, num_tensors: int) -> int:
    """Rows a frame always holds with `reserve` bytes reserved."""
    return max(0, (reserve - num_tensors * ALIGN) // row_bytes)


def _make_device():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def frame_rows_dev(frame_end, kept_top, row_bytes, rows, num_tensors: tl.constexpr):
        """Rows of `row_bytes` (summed over the frame's tensors) that fit between the kept
        top and frame_end, capped at `rows` (int64 scalars)."""
        fit = (frame_end - kept_top - num_tensors * 128) // row_bytes
        return tl.maximum(tl.minimum(rows, fit), 0)

    return frame_rows_dev


frame_rows = _make_device()
