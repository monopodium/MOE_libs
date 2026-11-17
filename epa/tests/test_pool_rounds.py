"""epa::Pool (pool.cuh) and epa.pool agree on sizing; epa.Rounds covers every row exactly once.
Run: python epa/tests/test_pool_rounds.py   (CPU; nvcc compiles the header's host side)"""
import itertools
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from epa import Rounds, pool  # noqa: E402

CASES = [(49152, 59392, 6), (30720, 32768, 3), (1, 128, 1), (12345, 14336, 5)]

src = '#include <epa/pool.cuh>\n#include <cstdio>\nint main() {\n'
for w, row, n in CASES:
    src += (f'  {{ epa::Pool p; p.reserve = epa::Pool::reserve_for({w}, {row}, {n});'
            f' printf("%lld %lld\\n", (long long)p.reserve, (long long)p.min_window({row}, {n})); }}\n')
src += '  return 0;\n}\n'
with tempfile.TemporaryDirectory() as d:
    (Path(d) / 't.cu').write_text(src)
    subprocess.run(['nvcc', '-std=c++17', f'-I{ROOT / "include"}', 't.cu', '-o', 't'], cwd=d, check=True)
    out = subprocess.run([str(Path(d) / 't')], capture_output=True, text=True, check=True).stdout.split('\n')
for (w, row, n), line in zip(CASES, out):
    reserve, mw = map(int, line.split())
    assert reserve == pool.reserve_bytes(w, row, n), (w, row, n)
    assert mw == pool.min_window(reserve, row, n) == w, (w, row, n, mw)

rng = random.Random(0)
for keys, window, max_rows in itertools.product([1, 7, 48], [1, 5, 300, 4096], [4096, 10000]):
    counts = [rng.randint(0, max_rows // keys) for _ in range(keys)]
    total = sum(counts)
    min_window = max(1, min(window, max_rows))
    rounds = Rounds(max_rows=max(total, 1), min_window=min_window)
    plan = rounds.plan(torch.tensor(counts, dtype=torch.int32), torch.tensor([min_window]))
    seen = [0] * keys
    for c in range(rounds.num_rounds):
        lo = c * min_window
        assert plan.start[c].item() == lo
        assert plan.rows[c].item() == plan.counts[c].sum().item() <= min_window
        for k in range(keys):
            seen[k] += plan.counts[c, k].item()
    assert seen == counts, (keys, window, counts, seen)
print(f'ok: Pool sizing agrees for {len(CASES)} cases; Rounds covers every row once')
