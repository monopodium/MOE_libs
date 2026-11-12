"""DeepEP V2.5 (direct NVLink dispatch): upstream vs EPArena, 8 GPUs.

Both modes run from the patched copy (DeepEP-epa); the upstream layout is used
when EP_EPA_CAP is unset, so the baseline is upstream's code path. For every
tokens-per-rank T, routing and dtype it reports
  buffer bytes (EPBuffer.num_bytes, the symmetric buffer DeepEP allocates)
  dispatch time  = dispatch_impl + dispatch_copy_epilogue_impl (median, profiler)
  combine time   = combine_impl + combine_reduce_epilogue_impl
  step time      = dispatch + combine with CUDA events
  bitwise equality of received tokens (order-free) and combined output vs upstream

usage: python bench_deepep.py --tokens 1024,4096,8192 --routing uniform,skew,extreme
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'DeepEP-epa'))
os.environ.setdefault('EP_DISABLE_GIN', '1')
os.environ.setdefault('EP_JIT_CACHE_DIR', '/run/user/1008/epa_dj_cache')
os.environ.setdefault('CUDA_HOME', '/home/Shuang/envs/cuda132')
os.environ['PATH'] = os.environ['CUDA_HOME'] + '/bin:' + os.environ['PATH']

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

H, E, K = 7168, 384, 6


def make_routing(T, kind, gen):
    if kind == 'uniform':
        scores = torch.rand(T, E, generator=gen, device='cuda')
    else:
        s_ = float(kind[4:]) if kind.startswith('zipf') else {'skew': 0.3, 'extreme': 1.0, 'hot': 1.5}[kind]   # 'zipf2.5': s = 2.5
        order = torch.randperm(E, generator=torch.Generator(device='cuda').manual_seed(7), device='cuda')
        logp = -s_ * torch.log(torch.arange(1, E + 1, device='cuda', dtype=torch.float))[order]
        g = -torch.log(-torch.log(torch.rand(T, E, generator=gen, device='cuda').clamp_min(1e-9)))
        scores = logp + g
    w, idx = scores.topk(K, dim=-1)
    return idx, torch.softmax(w, dim=-1).float()


ENV = ('EP_EPA_CAP', 'EP_EPA_POOL', 'EP_EPA_ARENA_P', 'EP_EPA_ARENA_PAGES')


class EnvMode:
    """EPArena is configured by env vars that DeepEP reads at sizing and at every dispatch.
    cap < -1 selects the epa::Arena: rows per page = -cap - 1, pool = pages."""
    def __init__(self, cap, pool):
        self.cap, self.pool = cap, pool

    def __enter__(self):
        for k in ENV:
            os.environ.pop(k, None)
        if self.cap is not None and self.cap < -1:
            os.environ['EP_EPA_ARENA_P'], os.environ['EP_EPA_ARENA_PAGES'] = str(-self.cap - 1), str(self.pool)
        elif self.cap is not None:
            os.environ['EP_EPA_CAP'], os.environ['EP_EPA_POOL'] = str(self.cap), str(self.pool)

    def __exit__(self, *a):
        for k in ENV:
            os.environ.pop(k, None)


def worker(rank, world, args):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    import deep_ep
    group = dist.new_group(list(range(world)))
    results = []
    for T in args.tokens:
        # expected tokens per (source, destination) pair and per receiver
        p_hit = 1 - math.comb(E - E // world, K) / math.comb(E, K)
        mean_pair = T * p_hit
        cfgs = [(None, None)] + [(math.ceil(a * mean_pair), math.ceil(b * world * mean_pair)) for a, b in args.tiers]
        # epa::Arena: pages of P tokens for beta x the mean receiver load, plus one partial page per source
        cfgs += [(-(int(P) + 1), math.ceil(b * world * mean_pair / P) + world) for P, b in args.arena]
        if args.upstream_last:
            cfgs = cfgs[1:] + cfgs[:1]
        for fp8 in args.fp8:
            bufs = {}
            for cap, pool in cfgs:
                with EnvMode(cap, pool):
                    bufs[(cap, pool)] = deep_ep.EPBuffer(group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
                                                         use_fp8_dispatch=bool(fp8), allow_hybrid_mode=False,
                                                         explicitly_destroy=True)
            for routing in args.routing:
                gen = torch.Generator(device='cuda').manual_seed(1000 * rank + T)
                topk, w = make_routing(T, routing, gen)
                topk = topk.to(deep_ep.topk_idx_t)
                x = torch.randn((T, H), generator=gen, device='cuda').to(torch.bfloat16)
                if fp8:
                    xa = x.float().view(T, H // 128, 128)
                    sf = (xa.abs().amax(-1).clamp_min(1e-4) / 448)
                    xq = (xa / sf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(T, H)
                    xin = (xq, sf.contiguous())
                else:
                    xin = x
                # per-(source, destination) token counts, for tier accounting
                dst = topk // (E // world)
                hit = torch.zeros(T, world, dtype=torch.int64, device='cuda').scatter_(1, dst.long(), 1)
                mine = hit.sum(0)
                allc = [torch.empty_like(mine) for _ in range(world)]
                dist.all_gather(allc, mine)
                pairs = torch.stack(allc)            # [src, dst]

                def run(buf, cap, pool, timing=None):
                    with EnvMode(cap, pool):
                        recv_x, _, recv_w, handle, _ = buf.dispatch(
                            x=xin, topk_idx=topk, topk_weights=w, num_experts=E, num_max_tokens_per_rank=T,
                            do_cpu_sync=False, allocate_on_comm_stream=False, async_with_compute_stream=False)
                        n = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
                        rx = recv_x[0] if fp8 else recv_x
                        y = torch.empty((rx.shape[0], H), dtype=torch.bfloat16, device='cuda')
                        y.copy_(rx.float())   # the "expert": identity (FP8 values without scales are fine for A/B)
                        out, _, _ = buf.combine(x=y, handle=handle, topk_weights=recv_w,
                                                allocate_on_comm_stream=False, async_with_compute_stream=False)
                    return rx, handle, n, out

                ref, checks = None, {}
                for (cap, pool), buf in sorted(bufs.items(), key=lambda kv: kv[0][0] is not None):
                    dist.barrier(); torch.cuda.synchronize()
                    rx, handle, n, out = run(buf, cap, pool)
                    torch.cuda.synchronize()
                    n = int(n)
                    src = handle.recv_src_metadata[:n, 0]
                    order = torch.argsort(src)
                    got = (src[order].clone(), rx[:n][order].view(torch.uint8).clone(), out.clone())
                    if ref is None:
                        ref, c = got, {}
                    else:
                        c = dict(dispatch_equal=bool(torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])),
                                 combine_equal=bool(torch.equal(ref[2], got[2])))
                    ok = torch.tensor([float(all(c.values()))], device='cuda')
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    c['all_ranks_equal'] = bool(ok.item())
                    checks[(cap, pool)] = c
                del ref, got
                # spill counts (arena mode): after each call, every receiver's `taken` table gives
                # the pages each source held; rows beyond taken * P went to the source's Home
                spills = {}
                for (cap, pool), buf in bufs.items():
                    if cap is None or cap >= -1 or not args.spill_calls:
                        continue
                    P = -cap - 1
                    per_call = []
                    for _ in range(args.spill_calls):
                        run(buf, cap, pool)
                        torch.cuda.synchronize()
                        with EnvMode(cap, pool):
                            have, got, pred = buf.runtime.epa_take_state(T).long()   # [dst] each, this source
                        held = torch.where(pred > have, have + got, pred)
                        assert torch.equal(pred, (pairs[rank] + P - 1) // P), 'pred != pages needed'
                        allt = [torch.empty_like(held) for _ in range(world)]
                        dist.all_gather(allt, held)
                        taken = torch.stack(allt)                   # [src, dst] pages held
                        sp = (pairs - taken * P).clamp_min(0)       # rows of (src -> dst) left in Home
                        per_call.append(dict(spilled_rows=int(sp.sum()), pairs_spilling=int((sp > 0).sum()),
                                             max_into_one_dst=int(sp.sum(0).max()),
                                             pages_used_max=int(taken.sum(0).max())))
                    spills[(cap, pool)] = dict(call_2=per_call[0], steady=per_call[-1],
                                               spilled_rows_all_calls=[c['spilled_rows'] for c in per_call])
                # timing: modes interleaved over reps
                samples = {m: [] for m in bufs}
                order = list(bufs.items())
                for rep in range(args.reps):
                    # rotate the order every rep so no mode always runs first
                    for (cap, pool), buf in order[rep % len(order):] + order[:rep % len(order)]:
                        for _ in range(3):
                            run(buf, cap, pool)
                        torch.cuda.synchronize(); dist.barrier()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            t0.record()
                            for _ in range(args.iters):
                                run(buf, cap, pool)
                            t1.record()
                            torch.cuda.synchronize()
                        kd = {'dispatch': [], 'combine': []}
                        for ev in prof.events():
                            if ev.device_type != torch.autograd.DeviceType.CUDA:
                                continue
                            if 'dispatch_impl' in ev.name or 'dispatch_copy_epilogue' in ev.name:
                                kd['dispatch'].append(ev.device_time_total)
                            elif 'combine_impl' in ev.name or 'combine_reduce_epilogue' in ev.name:
                                kd['combine'].append(ev.device_time_total)
                        # two kernels per op per iteration
                        d = sum(kd['dispatch']) / args.iters
                        cmb = sum(kd['combine']) / args.iters
                        samples[(cap, pool)].append((d, cmb, t0.elapsed_time(t1) * 1e3 / args.iters))
                for (cap, pool), buf in bufs.items():
                    v = torch.tensor([st.median(s[i] for s in samples[(cap, pool)]) for i in range(3)], device='cuda')
                    allv = [torch.empty_like(v) for _ in range(world)]
                    dist.all_gather(allv, v)
                    vmed = torch.stack(allv).median(0).values
                    beyond = (pairs - (cap if cap is not None and cap >= 0 else T)).clamp_min(0).sum().item()
                    mode = 'upstream' if cap is None else ('arena' if cap < -1 else 'epa')
                    rec = dict(T=T, routing=routing, fp8=fp8, mode=mode, cap=cap, pool=pool,
                               buffer_bytes=buf.num_bytes, dispatch_us=vmed[0].item(), combine_us=vmed[1].item(),
                               step_us=vmed[2].item(), beyond_cap_frac=beyond / max(1, pairs.sum().item()),
                               max_recv=int(pairs.sum(0).max()), mean_recv=pairs.sum().item() / world,
                               **checks[(cap, pool)], **({'spill': spills[(cap, pool)]} if (cap, pool) in spills else {}),
                               total_rows=int(pairs.sum()))
                    if rank == 0:
                        print(json.dumps(rec), flush=True)
                        results.append(rec)
            for b in bufs.values():
                b.destroy()
            del bufs
            torch.cuda.empty_cache()
            dist.barrier()
    if rank == 0 and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', default='1024,4096')
    p.add_argument('--routing', default='uniform,skew,extreme')
    p.add_argument('--arena', default='', help='epa::Arena modes, rows_per_page:beta,...')
    p.add_argument('--tiers', default='0:0,0:1.25,0.5:0.75',
                   help='alpha:beta; tier-1 slots = alpha * mean pair load, pool = beta * mean receiver load')
    p.add_argument('--fp8', default='0')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--port', type=int, default=29631)
    p.add_argument('--upstream-last', type=int, default=0)
    p.add_argument('--out', default='')
    p.add_argument('--spill-calls', type=int, default=0, help='arena modes: calls to record spill counts over')
    a = p.parse_args()
    a.tokens = [int(t) for t in a.tokens.split(',')]
    a.routing = a.routing.split(',')
    a.tiers = [tuple(float(v) for v in t.split(':')) for t in a.tiers.split(',')] if a.tiers else []
    a.arena = [tuple(float(v) for v in t.split(':')) for t in a.arena.split(',')] if a.arena else []
    a.fp8 = [int(t) for t in a.fp8.split(',')]
    mp.spawn(worker, args=(8, a), nprocs=8)
