"""Python mirror of epa/include/epa/window.cuh (Home + Window) for Python-orchestrated
EP libraries such as dist_moe. Same fields, same rule, same names.

    row r of the consumer's receive list is consumed in round r // shared at slot
    r % shared; rows past the window wait in their producer's Home.

Per-launch consumers run `num_rounds()` launches; launch `rnd` covers the rows that
`round_counts()` returns for it (zero rows: the launch is gated off on device).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math


def window_rows(beta: float, tokens: int, topk: int, align: int = 1) -> int:
    """Rows of a window that holds beta x the balanced load tokens * topk."""
    n = math.ceil(beta * tokens * topk)
    return -(-n // align) * align


@dataclasses.dataclass(frozen=True)
class Window:
    num_keys: int = 1   # consumer keys; 1 = unkeyed
    keyed: int = 0      # rows per key placed without a reservation
    shared: int = 0     # rows shared by all keys: W
    max_rows: int = 0   # bound on r for any routing: fixes the number of rounds

    @staticmethod
    def sized(beta: float, tokens: int, topk: int, max_rows: int, align: int = 1) -> Window:
        shared = min(window_rows(beta, tokens, topk, align), -(-max_rows // align) * align)
        return Window(shared=shared, max_rows=max_rows)

    def rows(self) -> int:
        return self.num_keys * self.keyed + self.shared

    def num_rounds(self) -> int:
        if self.shared <= 0:
            return 1
        return max(1, -(-self.max_rows // self.shared))

    def slot(self, r: int) -> int:
        return r % self.shared if self.shared > 0 else r

    def round(self, r: int) -> int:
        return r // self.shared if self.shared > 0 else 0

    def clip(self, begin: int, count: int, rnd: int) -> int:
        lo, hi = rnd * self.shared, (rnd + 1) * self.shared
        return max(0, min(begin + count, hi) - max(begin, lo))

    def overflow(self, n: int) -> int:
        return max(0, n - self.keyed)

    def accept(self, base: int, n: int) -> int:
        return base if base >= 0 and base + n <= self.shared else -1

    def row(self, key: int, j: int, base: int = -1) -> int:
        if j < self.keyed:
            return key * self.keyed + j
        if base < 0:
            return -1
        return self.num_keys * self.keyed + base + (j - self.keyed)

    def round_counts(self, counts):
        """Per-launch consumers: `counts` is a device tensor of per-key row counts in
        key-major order. Returns (counts_RE, rows_R, run_R): rows of each key in each
        round (clip() over every segment), rows per round, and an int32 flag per round
        (rows > 0). One fused device kernel, no host sync."""
        import torch

        n_rounds, n_keys = self.num_rounds(), counts.numel()
        out = torch.empty(n_rounds, n_keys + 2, dtype=torch.int32, device=counts.device)
        if counts.is_cuda:
            _round_counts_kernel[(n_rounds,)](counts, out, n_keys, self.shared,
                                              BLOCK=max(16, 1 << (n_keys - 1).bit_length()))
        else:
            end = counts.cumsum(0)
            lo = (torch.arange(n_rounds) * self.shared).unsqueeze(1)
            c = (torch.minimum(end, lo + self.shared) - torch.maximum(end - counts, lo)).clamp_min_(0)
            out[:, :n_keys] = c
            out[:, n_keys] = c.sum(1)
            out[:, n_keys + 1] = (c.sum(1) > 0).to(torch.int32)
        counts_RE = out[:, :n_keys]
        if counts_RE.dtype != counts.dtype:
            counts_RE = counts_RE.to(counts.dtype)
        return counts_RE, out[:, n_keys], out[:, n_keys + 1]


def _make_round_counts_kernel():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def kernel(counts_ptr, out_ptr, n_keys, shared, BLOCK: tl.constexpr):
        rnd = tl.program_id(0)
        k = tl.arange(0, BLOCK)
        c = tl.load(counts_ptr + k, mask=k < n_keys, other=0).to(tl.int64)
        end = tl.cumsum(c, 0)
        lo = rnd.to(tl.int64) * shared
        hi = lo + shared
        clipped = tl.maximum(tl.minimum(end, hi) - tl.maximum(end - c, lo), 0)
        row = out_ptr + rnd * (n_keys + 2)
        tl.store(row + k, clipped.to(tl.int32), mask=k < n_keys)
        total = tl.sum(clipped, 0)
        tl.store(row + n_keys, total.to(tl.int32))
        tl.store(row + n_keys + 1, (total > 0).to(tl.int32))

    return kernel


_round_counts_kernel = _make_round_counts_kernel()


@contextlib.contextmanager
def round_if_nonempty(rows_1):
    """Per-launch consumers: run one round's launches only if it has rows.

    Under CUDA-graph capture the body becomes a graph conditional (IF) node on the
    device scalar `rows_1 > 0`, so an empty round costs no launch. Eagerly it runs
    the body unconditionally. Yields True when the graph node does the gating (the
    kernels then need no in-kernel gate)."""
    import torch

    if not torch.cuda.is_current_stream_capturing():
        yield False
        return
    pred = (rows_1.reshape(()) > 0)
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    graph.begin_capture_to_if_node(pred)
    try:
        yield True
    finally:
        graph.end_capture_to_conditional_node()
