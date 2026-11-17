"""EPArena (EPA) for Python-orchestrated libraries.

Three ideas, one module each (C++ headers in include/epa/ hold the same rules):

  epa.window   Home + Window: rows a consumer has no room for wait in their producer's
               Home; row r runs in round r // W. Static W (host-sized).   window.cuh
  epa.rounds   the same rule with W chosen on the device per call, for consumers that
               launch once per round (grouped GEMMs).                      window.cuh
  epa.pool     one buffer for kept state (bottom, LIFO) and a per-call frame (top) that
               takes what is free; a reserve guarantees the frame a minimum window.
                                                                           pool.cuh
  epa.pull     rows dropped by a consumer are pulled again from their Home just before
               their last use (one read, instead of holding them or adding rounds).
                                                                           row_pull.cuh
Device allocators and cross-GPU memory (C++ first, Python wrappers where a library needs them):

  epa.arena    epa::Arena page ring: take / give pages from kernels.     arena.cuh
  epa.host     pinned-host tier: PagedWeights.                            host_tier.hpp
  epa.build    build helpers for the C++ extensions.

Usage by a library that runs a GEMM chain over received rows:

    from epa import Rounds, round_if, pool
    rounds = Rounds(max_rows=capacity_rows, min_window=pool.min_window(reserve, row_bytes, n))
    plan = rounds.plan(counts_per_expert, frame_rows_1)
    for c in range(rounds.num_rounds):
        with round_if(plan.rows_1(c)):
            ...
"""
from . import pool
from .pull import pull_rows
from .rounds import RoundPlan, Rounds, round_if
from .window import Window, round_if_nonempty, window_rows

__all__ = ['Window', 'window_rows', 'Rounds', 'RoundPlan', 'round_if', 'round_if_nonempty', 'pool', 'pull_rows']
