"""epa.rounds: the Window rule with a window chosen on the device, for per-launch consumers.

A per-launch consumer (a grouped GEMM that needs its rows contiguous) walks its key-major
receive list in rounds: round c covers rows [c * W, (c + 1) * W), each key's rows clipped
to that range. W is a device scalar chosen per call (e.g. the rows an `epa.pool` frame
holds this call), so the host never syncs. The number of launches is fixed when the
caller is built: ceil(max_rows / min_window), where min_window is the smallest W the
caller can ever be given. Rounds without rows are skipped (`round_if`).

    rounds = Rounds(max_rows=capacity, min_window=reserved_rows)
    plan = rounds.plan(counts_E, window_1)          # one tiny kernel
    for c in range(rounds.num_rounds):
        with round_if(plan.rows_1(c)):              # graph IF node when capturing
            gemm(counts=plan.counts[c], ptrs=plan.slice(ptrs, c),
                 rows_at=plan.offset(base_1, c, row_bytes), ...)
"""
from __future__ import annotations

import dataclasses

import torch

from .window import round_if_nonempty as round_if

__all__ = ['Rounds', 'RoundPlan', 'round_if']


@dataclasses.dataclass(frozen=True)
class RoundPlan:
    counts: torch.Tensor   # [K, keys] rows of each key per round; each row 16-byte aligned
    rows: torch.Tensor     # [K] int32 rows per round
    start: torch.Tensor    # [K] int64 first row of each round (c * W)

    def rows_1(self, c: int) -> torch.Tensor:
        """Device [1] rows of round c (the predicate of its IF node)."""
        return self.rows[c:c + 1]

    def offset(self, base_1: torch.Tensor, c: int, row_bytes: int) -> torch.Tensor:
        """Device [1] byte offset of round c's first row in an array that holds every row."""
        return base_1 + self.start[c:c + 1] * row_bytes

    def slice(self, src: torch.Tensor, c: int) -> torch.Tensor:
        """src[start_c:] as a fresh tensor of src's length (round 0: src itself).
        For per-row tables a consumer indexes from its first row (gather pointers)."""
        if c == 0:
            return src
        n = src.numel()
        idx = (self.start[c] + _arange(n, src.device)).clamp_(max=n - 1)
        return src.index_select(0, idx)


class Rounds:
    """Static launch count, device round plan."""

    def __init__(self, max_rows: int, min_window: int):
        if min_window <= 0:
            raise ValueError('min_window must be positive')
        self.max_rows = int(max_rows)
        self.min_window = int(min_window)
        self.num_rounds = max(1, -(-self.max_rows // self.min_window))

    def plan(self, counts: torch.Tensor, window_1: torch.Tensor) -> RoundPlan:
        """counts: device [keys] rows per key (key-major); window_1: device [1] W >= min_window.
        Rows past num_rounds * W are not covered: W >= min_window keeps that empty."""
        k, n_keys = self.num_rounds, counts.numel()
        pitch = -(-n_keys // 4) * 4   # each round's counts start on a 16-byte boundary
        per_key = torch.empty(k, pitch, dtype=torch.int32, device=counts.device)
        rows = torch.empty(k, dtype=torch.int32, device=counts.device)
        start = torch.empty(k, dtype=torch.int64, device=counts.device)
        if counts.is_cuda:
            _plan_kernel[(k,)](counts, window_1, per_key, rows, start, n_keys, pitch,
                               BLOCK=max(16, 1 << (n_keys - 1).bit_length()))
        else:
            w = int(window_1.reshape(()).item())
            end = counts.to(torch.int64).cumsum(0)
            lo = torch.arange(k, dtype=torch.int64).unsqueeze(1) * w
            c = (torch.minimum(end, lo + w) - torch.maximum(end - counts, lo)).clamp_min_(0)
            per_key[:, :n_keys] = c
            rows.copy_(c.sum(1))
            start.copy_(lo.squeeze(1))
        per_key = per_key[:, :n_keys]
        if per_key.dtype != counts.dtype:
            per_key = per_key.to(counts.dtype)
        return RoundPlan(counts=per_key, rows=rows, start=start)


_ARANGE: dict = {}


def _arange(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _ARANGE:
        _ARANGE[key] = torch.arange(n, dtype=torch.int64, device=device)
    return _ARANGE[key]


def _make_plan_kernel():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def kernel(counts_ptr, window_ptr, per_key_ptr, rows_ptr, start_ptr, n_keys, pitch,
               BLOCK: tl.constexpr):
        rnd = tl.program_id(0)
        k = tl.arange(0, BLOCK)
        c = tl.load(counts_ptr + k, mask=k < n_keys, other=0).to(tl.int64)
        end = tl.cumsum(c, 0)
        w = tl.load(window_ptr).to(tl.int64)
        lo = rnd.to(tl.int64) * w
        clipped = tl.maximum(tl.minimum(end, lo + w) - tl.maximum(end - c, lo), 0)
        tl.store(per_key_ptr + rnd * pitch + k, clipped.to(tl.int32), mask=k < n_keys)
        tl.store(rows_ptr + rnd, tl.sum(clipped, 0).to(tl.int32))
        tl.store(start_ptr + rnd, lo)

    return kernel


_plan_kernel = _make_plan_kernel()
