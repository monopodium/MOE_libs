# Page counts each library's take asks for, from the routing bench_deepep.py uses
# (E = 384, k = 6, 8 ranks, T = 4096). Writes one line per source rank.
#   deepep_<routing>.txt  [R][R]  pages per (source, destination rank), 64 rows per page
#   mc_<routing>.txt      [R][E]  pages per (source, expert), 16 rows per page (Mooncake arena)
import sys
import torch

R, E, K, T = 8, 384, 6, 4096
out = sys.argv[1]


def make_routing(kind, gen):
    if kind == 'uniform':
        scores = torch.rand(T, E, generator=gen)
    else:
        order = torch.randperm(E, generator=torch.Generator().manual_seed(7))
        logp = -1.0 * torch.log(torch.arange(1, E + 1, dtype=torch.float))[order]
        scores = logp - torch.log(-torch.log(torch.rand(T, E, generator=gen).clamp_min(1e-9)))
    return scores.topk(K, dim=-1).indices


for kind in ('uniform', 'extreme'):
    dp, mc = [], []
    for src in range(R):
        idx = make_routing(kind, torch.Generator().manual_seed(1000 + src))
        rows_e = torch.bincount(idx.flatten(), minlength=E)
        hit = torch.zeros(T, R, dtype=torch.bool)
        hit[torch.arange(T).unsqueeze(1), idx // (E // R)] = True
        tok_d = hit.sum(0)
        dp.append(((tok_d + 63) // 64).tolist())
        mc.append(((rows_e + 15) // 16).tolist())
    for name, m in (('deepep', dp), ('mc', mc)):
        with open(f'{out}/{name}_{kind}.txt', 'w') as f:
            f.write('\n'.join(' '.join(map(str, r)) for r in m) + '\n')
    print(kind, 'deepep pages per dst', [sum(r[d] for r in dp) for d in range(R)],
          'mc pages per src', [sum(r) for r in mc], 'mc max key', max(max(r) for r in mc))
