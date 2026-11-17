"""Compare two dumps of epa_train.py (every rank): bitwise equality and max relative error."""
import sys
from pathlib import Path

import torch


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


ref, new = Path(sys.argv[1]), Path(sys.argv[2])
worst = {}
equal = {}
for f in sorted(ref.glob('rank*.pt')):
    r, n = torch.load(f), torch.load(new / f.name)
    pairs = [('out', r['out'], n['out']), ('grad_x', r['gx'], n['gx']), ('grad_scores', r['gs'], n['gs'])]
    for i, ((a13, a2), (b13, b2)) in enumerate(zip(r['gw'], n['gw'])):
        pairs += [(f'grad_w13[{i}]', a13, b13), (f'grad_w2[{i}]', a2, b2)]
    for name, a, b in pairs:
        equal[name] = equal.get(name, True) and a['hash'] == b['hash']
        worst[name] = max(worst.get(name, 0.0), rel(b['sample'], a['sample']))
for name in worst:
    print(f'{name:14s} bitwise={equal[name]!s:5s} max_rel(1M-sample)={worst[name]:.3e}')
