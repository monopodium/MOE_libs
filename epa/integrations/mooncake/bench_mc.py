"""Mooncake EP legacy (low-latency) kernels: upstream vs EPArena, 8 GPUs over NVLink.

For each tokens-per-rank T and routing, it runs
  upstream                 Mooncake's BufferPair layout and kernels, unmodified
  epa cap=c                EPArena layout, tier-1 slots `c` per (expert, source)
and reports heap bytes, dispatch/combine latency (CUDA events, median of N), the
fraction of messages that went to tier 3, and a bitwise check of epa vs upstream.

usage: python bench_mc.py [--tokens 128,256,...] [--routing uniform,zipf] [--fp8 0]
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'python'))
import epa_build  # noqa: E402

MC = Path('/home/Shuang/shuangma/mkernel_project/MOE_libs/Mooncake')
H, E, K = 7168, 384, 6


def build():
    return epa_build.load('epa_mooncake', [HERE / 'harness.cu'],
                          includes=[HERE, MC / 'mooncake-ep/include', MC / 'mooncake-transfer-engine/include'],
                          defines=['USE_CUDA'])


def make_routing(T, kind, gen):
    if kind == 'uniform':
        scores = torch.rand(T, E, generator=gen, device='cuda')
    else:
        # Zipf expert popularity in a fixed random expert order (same on every rank),
        # sampled with Gumbel top-k. s = 0.3 ('skew') puts ~4x the mean load on the
        # hottest expert, close to measured DeepSeek-style imbalance; s = 1.0
        # ('extreme') puts ~40x, an adversarial case.
        s_ = {'skew': 0.3, 'extreme': 1.0}[kind]
        order = torch.randperm(E, generator=torch.Generator(device='cuda').manual_seed(7), device='cuda')
        logp = -s_ * torch.log(torch.arange(1, E + 1, device='cuda', dtype=torch.float))[order]
        g = -torch.log(-torch.log(torch.rand(T, E, generator=gen, device='cuda').clamp_min(1e-9)))
        scores = logp + g
    w, idx = scores.topk(K, dim=-1)
    return idx.to(torch.int64), torch.softmax(w, dim=-1).float()


class Inst:
    def __init__(self, ext, epa, T, cap, pool, fp8, rank, world):
        self.ext = ext
        self.id, self.bytes = ext.create(epa, T, H, E, K, cap, pool, fp8, rank, world)
        mine = ext.handle(self.id)
        allh = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather_object(allh, mine)
        ext.open(self.id, torch.stack(allh))
        torch.cuda.synchronize()
        dist.barrier()

    def y(self, y):
        return y

    def dispatch(self, x, topk, o):
        self.ext.dispatch(self.id, x, topk, o['recv_x'], o['scales'], o['src_info'], o['layout'], o['count'])

    def combine(self, y, topk, w, o, out):
        self.ext.combine(self.id, y, topk, w, o['src_info'], o['layout'], out)

    def stats(self):
        return self.ext.stats(self.id)

    def arena_stats(self):
        return self.ext.arena_stats(self.id)

    def destroy(self):
        self.ext.destroy(self.id)


class PullInst(Inst):
    """EPA-pull: no receive buffers; y lives in the symmetric heap."""
    def __init__(self, ext, T, fp8, rank, world, blocks):
        self.ext = ext
        self.id, self.bytes, self.total = ext.pull_create(T, H, E, K, fp8, rank, world, blocks)
        mine = ext.pull_handle(self.id)
        allh = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather_object(allh, mine)
        ext.pull_open(self.id, torch.stack(allh))
        self.ytensor = ext.pull_y(self.id)
        torch.cuda.synchronize()
        dist.barrier()

    def y(self, y):
        return self.ytensor

    def dispatch(self, x, topk, o):
        self.ext.pull_dispatch(self.id, x, topk, o['recv_x'], o['scales'], o['src_info'], o['layout'], o['count'])

    def combine(self, y, topk, w, o, out):
        self.ext.pull_combine(self.id, topk, w, out)

    def stats(self):
        return [0, 0]

    def destroy(self):
        self.ext.pull_destroy(self.id)


def outputs(T, R, fp8):
    L = E // R
    o = dict(recv_x=torch.empty((L, R * T, H), dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16, device='cuda'),
             src_info=torch.empty((L, R * T), dtype=torch.int32, device='cuda'),
             layout=torch.empty((L, R), dtype=torch.int64, device='cuda'),
             count=torch.empty((L,), dtype=torch.int32, device='cuda'))
    o['scales'] = (torch.empty((L, R * T, H // 128), dtype=torch.float, device='cuda').transpose(1, 2).contiguous().transpose(1, 2)
                   if fp8 else None)
    return o


def run_pair(ext, inst, x, topk, w, o, y, out):
    """One decode-step's worth of EP traffic: dispatch then combine."""
    inst.dispatch(x, topk, o)
    inst.combine(inst.y(y), topk, w, o, out)


def canonical(o, R, T):
    """Rows of every local expert sorted by (source rank, token): order-free view."""
    L = E // R
    lay = o['layout'].cpu()
    cnt, beg = (lay & 0xffffffff).to(torch.int64), (lay >> 32).to(torch.int64)   # pack2(count, begin)
    rows, keys = [], []
    info = o['src_info'].cpu()
    for e in range(L):
        for s in range(R):
            n, b = int(cnt[e, s]), int(beg[e, s])
            if n == 0:
                continue
            tok = info[e, b:b + n] & 0xffffff
            order = torch.argsort(tok)
            keys.append(torch.stack([torch.full((n,), e), torch.full((n,), s), tok[order]], 1))
            rows.append(o['recv_x'][e, b:b + n][order.cuda()])
    return torch.cat(keys), torch.cat(rows)


def profile_graph(g, iters):
    """Replay `g` once under the profiler: per-kernel median durations and pair time."""
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        t0.record(); g.replay(); t1.record()
        torch.cuda.synchronize()
    kd = {'dispatch': [], 'combine': []}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ('mooncake' in ev.name or 'epa_pull' in ev.name):
            for k in kd:
                if k in ev.name:
                    kd[k].append(ev.device_time_total)
    return st.median(kd['dispatch']), st.median(kd['combine']), t0.elapsed_time(t1) * 1e3 / iters


def worker(rank, world, args):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    ext = build()
    results = []
    L = E // world
    for T in args.tokens:
        mean_pair = T * K / E
        # (tier-1 slots per pair, tier-2 pool rows): alpha * mean pair load, beta * mean receiver load
        cfgs = sorted({(max(0, math.ceil(a * mean_pair)), math.ceil(b * T * K)) for a, b in args.tiers} |
                      ({(T, 0)} if args.full_cap else set()))
        for routing in args.routing:
            gen = torch.Generator(device='cuda').manual_seed(1000 * rank + T)
            topk, w = make_routing(T, routing, gen)
            x = (torch.randn((T, H), generator=gen, device='cuda') * 0.5).to(torch.bfloat16)
            # per-(source, expert) loads, for the tier-3 fraction
            pair_counts = torch.zeros(E, dtype=torch.int64, device='cuda')
            pair_counts.scatter_add_(0, topk.flatten(), torch.ones(T * K, dtype=torch.int64, device='cuda'))
            all_pairs = [torch.empty_like(pair_counts) for _ in range(world)]
            dist.all_gather(all_pairs, pair_counts)
            pairs = torch.stack(all_pairs)
            for fp8 in args.fp8:
                modes = ([('upstream', None, None)] if T <= args.max_upstream_T else []) + [('epa', c, b) for c, b in cfgs]
                modes += [('pull', None, None)] if args.pull else []
                # epa::Arena: (rows per page, arena rows = beta * T * K)
                modes += [('arena', int(p_), math.ceil(b_ * T * K)) for p_, b_ in args.arena]
                insts = {m: (PullInst(ext, T, bool(fp8), rank, world, args.pull_blocks) if m[0] == 'pull' else
                             Inst(ext, {'upstream': 0, 'epa': 1, 'arena': 2}[m[0]], T, m[1] or 0, m[2] or 0,
                                  bool(fp8), rank, world)) for m in modes}
                o = outputs(T, world, bool(fp8))
                y = torch.empty((L, world * T, H), dtype=torch.bfloat16, device='cuda')
                out = torch.empty((T, H), dtype=torch.bfloat16, device='cuda')
                scale = (1 + torch.arange(L, device='cuda', dtype=torch.float) / 64).view(-1, 1, 1)
                checks, ref = {}, None
                # correctness: one dispatch + combine per mode; the "expert" is y = recv_x * (1 + e/64)
                for m, inst in insts.items():
                    torch.cuda.synchronize(); dist.barrier()
                    inst.dispatch(x, topk, o)
                    torch.cuda.synchronize(); dist.barrier()
                    yy = inst.y(y)
                    for e_ in range(L):   # one expert at a time keeps the FP32 temporary small
                        rx = o['recv_x'][e_].float() * (o['scales'][e_].repeat_interleave(128, dim=1) if fp8 else 1)
                        yy[e_].copy_((rx * scale[e_]).to(torch.bfloat16))
                    del rx
                    inst.combine(inst.y(y), topk, w, o, out)
                    torch.cuda.synchronize(); dist.barrier()
                    tiers = torch.tensor(inst.stats(), device='cuda', dtype=torch.float)
                    dist.all_reduce(tiers)
                    keys, rows = canonical(o, world, T)
                    el = (topk % L).float()
                    refy = (x.float().unsqueeze(1) * (1 + el / 64).unsqueeze(-1)).to(torch.bfloat16).float()
                    c = {'combine_max_abs_err': (out.float() - (refy * w.unsqueeze(-1)).sum(1)).abs().max().item(),
                         'tier2_frac': tiers[0].item() / (world * T * K), 'tier3_frac': tiers[1].item() / (world * T * K)}
                    if ref is None:
                        ref = (keys, rows, out.clone())
                    else:
                        c['dispatch_equal'] = bool(torch.equal(ref[0], keys) and
                                                   torch.equal(ref[1].view(torch.uint8), rows.view(torch.uint8)))
                        c['combine_equal'] = bool(torch.equal(ref[2], out))
                    ok = torch.tensor([float(all(v for k_, v in c.items() if k_.endswith('equal')))], device='cuda')
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    c['all_ranks_equal'] = bool(ok.item())
                    checks[m] = c
                    del keys, rows
                del ref
                # performance: each mode's pairs captured in a graph; modes measured round-robin
                graphs = {}
                for m, inst in insts.items():
                    for _ in range(3):
                        run_pair(ext, inst, x, topk, w, o, y, out)
                    torch.cuda.synchronize(); dist.barrier()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(args.iters):
                            run_pair(ext, inst, x, topk, w, o, y, out)
                    torch.cuda.synchronize(); dist.barrier()
                    g.replay(); torch.cuda.synchronize(); dist.barrier()
                    graphs[m] = g
                samples = {m: [] for m in insts}
                for _ in range(args.reps):
                    for m in insts:
                        dist.barrier()
                        samples[m].append(profile_graph(graphs[m], args.iters))
                for m, inst in insts.items():
                    v = torch.tensor([st.median(s_[i] for s_ in samples[m]) for i in range(3)], device='cuda')
                    vmax, vmed = v.clone(), v.clone()
                    dist.all_reduce(vmax, op=dist.ReduceOp.MAX)
                    allv = [torch.empty_like(v) for _ in range(world)]
                    dist.all_gather(allv, v)
                    vmed = torch.stack(allv).median(0).values
                    cap, pool = m[1], m[2]
                    tier23 = (pairs - cap).clamp_min(0).sum().item() if m[0] == 'epa' else 0
                    ar = torch.tensor(inst.arena_stats() if m[0] == 'arena' else [0, 0, 0, 0], device='cuda')
                    dist.all_reduce(ar, op=dist.ReduceOp.MAX)
                    if m[0] == 'arena':
                        cap = pool = None
                    rec = dict(T=T, routing=routing, fp8=fp8, mode=m[0], cap=cap, pool=pool, heap_bytes=inst.bytes,
                               arena_rows_per_page=m[1] if m[0] == 'arena' else None,
                               arena_beta=m[2] / (T * K) if m[0] == 'arena' else None,
                               arena_pages=int(ar[0]), arena_peak_pages=int(ar[1]), arena_page_bytes=int(ar[2]),
                               arena_failed_takes=int(ar[3]),
                               dispatch_us=vmed[0].item(), combine_us=vmed[1].item(), pair_us=vmed[2].item(),
                               dispatch_us_max=vmax[0].item(), combine_us_max=vmax[1].item(),
                               beyond_cap_frac=tier23 / (world * T * K), max_pair=int(pairs.max()), mean_pair=mean_pair,
                               max_recv=int(pairs.view(world, world, -1).sum((0, 2)).max()), mean_recv=T * K,
                               packed_bytes=sum(t.numel() * t.element_size() for t in o.values() if t is not None),
                               **checks[m])
                    if rank == 0:
                        print(json.dumps(rec), flush=True)
                        results.append(rec)
                del graphs
                torch.cuda.synchronize(); dist.barrier()
                for inst in insts.values():
                    inst.destroy()
                del o, y, out, insts
                torch.cuda.empty_cache()
                dist.barrier()
    if rank == 0 and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', default='128,256,512,1024')
    p.add_argument('--routing', default='uniform,skew,extreme')
    p.add_argument('--tiers', default='0:0,1:0,1:1,2:0.5,4:0',
                   help='alpha:beta pairs; tier-1 slots = alpha * T*K/E per pair, tier-2 pool = beta * T*K rows')
    p.add_argument('--full-cap', type=int, default=1, help='also run epa with cap = T (no tier 3)')
    p.add_argument('--fp8', default='0')
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--max-upstream-T', type=int, default=4096)
    p.add_argument('--world', type=int, default=8)
    p.add_argument('--port', type=int, default=29621)
    p.add_argument('--pull', type=int, default=0, help='also run EPA-pull')
    p.add_argument('--pull-blocks', type=int, default=0, help='CTAs for EPA-pull kernels (0: all SMs)')
    p.add_argument('--arena', default='', help='epa::Arena modes, rows_per_page:beta,...')
    p.add_argument('--out', default='')
    a = p.parse_args()
    a.tokens = [int(t) for t in a.tokens.split(',')]
    a.routing = a.routing.split(',')
    a.tiers = [tuple(float(v) for v in t.split(':')) for t in a.tiers.split(',')]
    a.arena = [tuple(float(v) for v in t.split(':')) for t in a.arena.split(',')] if a.arena else []
    a.fp8 = [int(t) for t in a.fp8.split(',')]
    build()
    mp.spawn(worker, args=(a.world, a), nprocs=a.world)
