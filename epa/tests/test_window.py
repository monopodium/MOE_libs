"""epa::Window (C++ header, host side) and epa_window.Window (Python) agree on every
rule the integrations use. Run: python epa/tests/test_window.py"""
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from epa_window import Window, window_rows  # noqa: E402

CASES = [  # num_keys, keyed, shared, max_rows
    (1, 0, 49152, 196608), (1, 0, 6144, 6264), (8, 512, 3072, 32768),
    (384, 4, 1024, 384 * 128), (48, 8192, 0, 8192), (1, 0, 7, 50),
]
RS = [0, 1, 6, 7, 511, 512, 6143, 6144, 6264, 49151, 49152, 100000]


def cpp_values():
    src = ['#include <cstdio>', '#include <epa/window.cuh>', 'int main() {']
    src.append('printf("%lld %lld %lld\\n", (long long)epa::window_rows(1.25, 4096, 6), '
               '(long long)epa::window_rows(2.0, 1000, 6, 128), (long long)epa::window_rows(0.3, 7, 3));')
    for nk, kd, sh, mx in CASES:
        src.append(f'{{ epa::Window w; w.num_keys={nk}; w.keyed={kd}; w.shared={sh}; w.max_rows={mx};')
        src.append('printf("%lld %lld\\n", (long long)w.rows(), (long long)w.num_rounds());')
        for r in RS:
            if sh > 0:
                src.append(f'printf("%lld %lld\\n", (long long)w.slot({r}), (long long)w.round({r}));')
            for rnd in range(3):
                src.append(f'printf("%lld\\n", (long long)w.clip({r}, 1000, {rnd}));')
            src.append(f'printf("%lld %lld\\n", (long long)w.overflow({r}), (long long)w.accept({r}, 100));')
            for key, base in itertools.product([0, 3], [-1, 0, 17]):
                src.append(f'printf("%lld\\n", (long long)w.row({key}, {r}, {base}));')
        src.append('}')
    src.append('}')
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 't.cpp').write_text('\n'.join(src))
        subprocess.check_call(['g++', '-std=c++17', '-I', str(ROOT / 'include'), 't.cpp', '-o', 't'], cwd=d)
        return subprocess.check_output([str(Path(d) / 't')], text=True).split()


def py_values():
    out = [window_rows(1.25, 4096, 6), window_rows(2.0, 1000, 6, 128), window_rows(0.3, 7, 3)]
    for nk, kd, sh, mx in CASES:
        w = Window(nk, kd, sh, mx)
        out += [w.rows(), w.num_rounds()]
        for r in RS:
            if sh > 0:
                out += [w.slot(r), w.round(r)]
            out += [w.clip(r, 1000, rnd) for rnd in range(3)]
            out += [w.overflow(r), w.accept(r, 100)]
            out += [w.row(key, r, base) for key, base in itertools.product([0, 3], [-1, 0, 17])]
    return [str(v) for v in out]


def test_round_counts():
    import torch
    w = Window(shared=10, max_rows=40)
    counts = torch.tensor([3, 12, 0, 9, 6], dtype=torch.int32)
    rc, rows, run = w.round_counts(counts)
    begins = [0, 3, 15, 15, 24]
    ref = [[w.clip(b, int(c), rnd) for b, c in zip(begins, counts.tolist())] for rnd in range(w.num_rounds())]
    assert rc.tolist() == ref, (rc.tolist(), ref)
    assert rc.sum().item() == counts.sum().item()
    assert rows.tolist() == [sum(r) for r in ref] and run.tolist() == [int(sum(r) > 0) for r in ref]
    if torch.cuda.is_available():
        g = [t.cpu().tolist() for t in w.round_counts(counts.cuda())]
        assert g == [rc.tolist(), rows.tolist(), run.tolist()], g


if __name__ == '__main__':
    c, p = cpp_values(), py_values()
    assert c == p, next((i, a, b) for i, (a, b) in enumerate(zip(c, p)) if a != b)
    test_round_counts()
    print(f'ok: {len(c)} values agree between window.cuh and epa_window.py; round_counts ok')
