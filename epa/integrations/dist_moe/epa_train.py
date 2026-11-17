"""DistMOE training step (L chained MoE layers, forward + backward) on upstream or the EPA fork.

torchrun --nproc-per-node 8 epa_train.py --pkg fork --epa 2.0 --routing uniform --slot min --layers 4
Writes one JSON line (rank 0) and, with --dump DIR, every rank's outputs and gradients of step 0.
"""
import argparse
import json
import os
import statistics as st
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = {'upstream': os.environ.get('DIST_MOE_UPSTREAM', os.path.join(_HERE, '../../../dist_moe')),
         'fork': os.path.join(_HERE, 'dist_moe-epa')}
H, I, E, K = 7168, 3072, 384, 6


def routing(T, kind, gen):
    if kind == 'uniform':
        scores = torch.rand(T, E, generator=gen, device='cuda')
    else:
        s_ = {'skew': 0.3, 'extreme': 1.0}[kind]
        order = torch.randperm(E, generator=torch.Generator(device='cuda').manual_seed(7), device='cuda')
        logp = -s_ * torch.log(torch.arange(1, E + 1, device='cuda', dtype=torch.float))[order]
        scores = logp - torch.log(-torch.log(torch.rand(T, E, generator=gen, device='cuda').clamp_min(1e-9)))
    w, idx = scores.topk(K, dim=-1)
    return idx.to(torch.int64), torch.softmax(w, dim=-1).float()


def fingerprint(t, samples=1 << 20):
    """Exact hash of t's bits plus a fixed sample of its values (a few MB instead of GBs)."""
    flat = t.detach().contiguous().view(-1)
    bits = flat.view(torch.int16 if flat.element_size() == 2 else torch.int32).to(torch.int64)
    pos = torch.arange(bits.numel(), device=bits.device, dtype=torch.int64) % 1000003 + 1
    idx = torch.randint(0, flat.numel(), (min(samples, flat.numel()),),
                        generator=torch.Generator(device=flat.device).manual_seed(0), device=flat.device)
    return {'hash': (int((bits * pos).sum().item()), int(bits.sum().item()), flat.numel()),
            'sample': flat[idx].float().cpu()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='fork')
    ap.add_argument('--epa', type=float, default=0.0, help='window factor; 0 = off')
    ap.add_argument('--routing', default='uniform')
    ap.add_argument('--slot', default='min', help="'min' (keep x only) or a slot capacity factor")
    ap.add_argument('--no-lean', action='store_true')
    ap.add_argument('--pool', type=float, default=0.0, help='EPA pool GiB per GPU (replaces --slot)')
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--factor', type=float, default=8.0)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--dump', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--graph', action='store_true', help='time replays of one CUDA graph of the whole step')
    a = ap.parse_args()
    sys.path.insert(0, ROOTS[a.pkg])
    import dist_moe

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    group = dist.group.WORLD
    R = dist.get_world_size()
    T, L = a.tokens, a.layers
    gen = torch.Generator(device='cuda').manual_seed(100 + rank)
    x0 = torch.randn(T, H, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.1
    topk, scores0 = routing(T, a.routing, gen)
    ws = [(torch.randn(E // R, 2 * I, H, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.02,
           torch.randn(E // R, H, I, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.02) for _ in range(L)]
    ws = [(torch.nn.Parameter(a_), torch.nn.Parameter(b_)) for a_, b_ in ws]
    gout = torch.randn(T, H, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.01
    recv = torch.bincount(topk.flatten() // (E // R), minlength=R)
    dist.all_reduce(recv)

    kw = dict(max_num_local_input_tokens=T, hidden_dim=H, intermediate_dim=I, top_k=K, num_experts=E,
              max_moe_layers_per_activation_slot=L, device_scratch_capacity_factor=a.factor)
    if a.slot != 'min' and not a.pool:
        kw['activation_slot_capacity_factor'] = float(a.slot)
    if a.epa > 0:
        pool = dict(pool_bytes=int(a.pool * 2**30)) if a.pool else {}
        kw['epa'] = dist_moe.api.EpaConfig(window_factor=a.epa, lean=not a.no_lean, **pool)
    config = dist_moe.Config(**kw)
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    ctx = dist_moe.create_context(group=group, config=config, device=torch.device('cuda', rank))
    torch.cuda.synchronize()
    m_ctx = torch.cuda.memory_allocated() - m0
    plan = ctx.memory_plan

    def step():
        ctx.reset()
        x = x0.detach().clone().requires_grad_()
        scores = scores0.detach().clone().requires_grad_()
        out = x
        for w13, w2 in ws:
            out = out + dist_moe.routed_experts(out, topk, scores, w13, w2, ctx)
        out.backward(gout)
        return out, x, scores

    for p in (p for pair in ws for p in pair):
        p.grad = None
    out, x, scores = step()
    torch.cuda.synchronize()
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        torch.save({'out': fingerprint(out.detach()), 'gx': fingerprint(x.grad), 'gs': fingerprint(scores.grad),
                    'gw': [(fingerprint(w13.grad), fingerprint(w2.grad)) for w13, w2 in ws]}, f'{a.dump}/rank{rank}.pt')
    run = step
    if a.graph:
        # fresh parameters: their AccumulateGrad nodes must first run on the capture's stream
        ws[:] = [(torch.nn.Parameter(w13.detach()), torch.nn.Parameter(w2.detach())) for w13, w2 in ws]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(a.warmup):
                for p in (p for pair in ws for p in pair):
                    p.grad = None
                step()
        torch.cuda.current_stream().wait_stream(side)
        for p in (p for pair in ws for p in pair):
            p.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        run = graph.replay
        run()
    else:
        for _ in range(a.warmup):
            for p in (p for pair in ws for p in pair):
                p.grad = None
            step()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(a.iters):
        if not a.graph:
            for p in (p for pair in ws for p in pair):
                p.grad = None
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        run()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1000)
    if os.environ.get('EPA_PROF') and rank == 0:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25))
    elif os.environ.get('EPA_PROF'):
        step(); torch.cuda.synchronize()
    peak_step = torch.cuda.max_memory_allocated() - base
    med = torch.tensor([st.median(times)], device='cuda')
    dist.all_reduce(med, op=dist.ReduceOp.MAX)
    stats = torch.tensor([m_ctx, peak_step], device='cuda', dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        rec = dict(tag=a.tag, graph=a.graph, pkg=a.pkg, epa=a.epa, pool=a.pool, lean=not a.no_lean, routing=a.routing, slot=a.slot, layers=L, T=T, R=R,
                   factor=a.factor, step_us=round(med.item(), 1), times_us=[round(t, 1) for t in times],
                   context_bytes=int(stats[0].item()), step_peak_bytes=int(stats[1].item()),
                   buffer_bytes=plan.total_device_buffer_bytes, scratch_bytes=plan.device_scratch_bytes,
                   slot_bytes=plan.activation_slot_bytes, window_rows=plan.device_scratch_capacity_rows,
                   capacity_rows=plan.total_scratch_capacity_rows, max_recv_rows=int(recv.max().item()))
        print('RESULT', json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
    ctx.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
