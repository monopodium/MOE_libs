"""dist_moe (BF16 inference): worst-case scratch capacity vs EPArena-sized capacity.

dist_moe sizes its per-rank activation scratch as F * T * k received rows
(`device_scratch_capacity_factor`). F = 8 (= R) is the only value safe for every
routing; the default F = 1 traps when a rank receives more than T*k rows.
EPArena's sizing rule is F = the receiver-level imbalance (EPLB-bounded).

One process group per (F, routing), launched by `run` below with torchrun:
reports the context's device memory, routed_experts time (CUDA events), the
received rows / capacity ratio, and whether it trapped.
"""
import json
import os
import statistics as st
import sys

import torch
import torch.distributed as dist

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


def main():
    import dist_moe
    T, F, kind, iters = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    group = dist.group.WORLD
    R = dist.get_world_size()
    gen = torch.Generator(device='cuda').manual_seed(100 + rank)
    x = torch.randn(T, H, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.1
    topk, w = routing(T, kind, gen)
    w13 = torch.randn(E // R, 2 * I, H, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.02
    w2 = torch.randn(E // R, H, I, dtype=torch.bfloat16, device='cuda', generator=gen) * 0.02
    # rows this rank will receive, and the largest over ranks
    recv = torch.bincount(topk.flatten() // (E // R), minlength=R)
    dist.all_reduce(recv)
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    config = dist_moe.Config(max_num_local_input_tokens=T, hidden_dim=H, intermediate_dim=I, top_k=K,
                             num_experts=E, max_moe_layers_per_activation_slot=1,
                             device_scratch_capacity_factor=F, activation_slot_capacity_factor=None,
                             num_activation_slots=0, inference=True)
    ctx = dist_moe.create_context(group=group, config=config, device=torch.device('cuda', rank))
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    plan = dist_moe.plan_memory(config, R, device=torch.device('cuda', rank))
    rec = dict(T=T, F=F, routing=kind, ctx_bytes=m1 - m0, max_recv_rows=int(recv.max()), mean_recv_rows=T * K,
               capacity_rows=getattr(plan, 'device_scratch_capacity_rows', None))
    graph = os.environ.get('BENCH_GRAPH') == '1'
    with torch.no_grad():
        for _ in range(3):
            out = dist_moe.routed_experts(x, topk, w, w13, w2, ctx)
        torch.cuda.synchronize()
        if graph:
            # inference serving replays a captured graph: no host launch cost
            g = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(g):
                out = dist_moe.routed_experts(x, topk, w, w13, w2, ctx)
            torch.cuda.synchronize()
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            dist.barrier()
            a.record()
            if graph:
                g.replay()
            else:
                out = dist_moe.routed_experts(x, topk, w, w13, w2, ctx)
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
    if os.environ.get('BENCH_PROFILE') == '1':
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                dist.barrier()
                dist_moe.routed_experts(x, topk, w, w13, w2, ctx)
            torch.cuda.synchronize()
        if rank == 0:
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=70), flush=True)
    t = torch.tensor([st.median(ts)], device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    rec['us'] = t.item()
    rec['graph'] = graph
    rec['epa_chunk'] = os.environ.get('DIST_MOE_EPA_CHUNK')
    rec['checksum'] = float(out.float().abs().sum())
    if rank == 0:
        print('RESULT ' + json.dumps(rec), flush=True)
    dist_moe.close_context(ctx) if hasattr(dist_moe, 'close_context') else None
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
