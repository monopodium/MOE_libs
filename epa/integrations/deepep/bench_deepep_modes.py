"""DeepEP V2.5 direct NVLink EP, 8 GPUs: upstream vs epa::Arena across the call modes.

DeepEP V2 has no separate low-latency API: decoding uses the same EPBuffer with the
expanded layout (do_expand=True, one row per token x local expert) and GPU-side counts
(do_cpu_sync=False). This bench sweeps
  --expand    0: normal layout (one row per token per rank), 1: expanded (the decode / LL layout)
  --cpu-sync  0: outputs sized for the worst case, 1: host waits for the counts, exact sizes
and reports for each mode
  buffer_bytes     the symmetric EP buffer (EPBuffer.num_bytes)
  recv_x_bytes     the dispatch output as allocated, and the rows that hold data
  peak_tensor      PyTorch peak over one dispatch + expert copy + combine, above what was live before
  dispatch/combine kernel time (profiler), step time (CUDA events) and host wall time per step
  equality vs upstream: received rows (order-free checksums) and combined output

usage: python bench_deepep_modes.py --tokens 128,256 --expand 1 --cpu-sync 0,1 --arena 64:1.25,16:1.25
"""
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

from bench_deepep import H, E, K, EnvMode, make_routing, torch, dist, mp


def row_sums(rx, n):
    """Order-free fingerprint of the first n rows: per-row int64 checksums, sorted."""
    v = rx[:n].contiguous().view(torch.int32).long()
    w = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=torch.long)
    return torch.sort((v * w).sum(1)).values


def worker(rank, world, args):
    import os
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    import deep_ep
    group = dist.new_group(list(range(world)))
    results = []
    for T in args.tokens:
        p_hit = 1 - math.comb(E - E // world, K) / math.comb(E, K)
        mean_pair = T * p_hit
        cfgs = [(None, None)] + [(-(int(P) + 1), math.ceil(b * world * mean_pair / P) + world) for P, b in args.arena]
        bufs = {}
        for cap, pool in cfgs:
            with EnvMode(cap, pool):
                bufs[(cap, pool)] = deep_ep.EPBuffer(group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
                                                     allow_hybrid_mode=False, explicitly_destroy=True)
        for routing in args.routing:
            gen = torch.Generator(device='cuda').manual_seed(1000 * rank + T)
            topk, w = make_routing(T, routing, gen)
            topk = topk.to(deep_ep.topk_idx_t)
            x = torch.randn((T, H), generator=gen, device='cuda').to(torch.bfloat16)
            dst = topk // (E // world)
            hit = torch.zeros(T, world, dtype=torch.int64, device='cuda').scatter_(1, dst.long(), 1)
            allc = [torch.empty_like(hit.sum(0)) for _ in range(world)]
            dist.all_gather(allc, hit.sum(0))
            pairs = torch.stack(allc)            # [src, dst] tokens

            for expand in args.expand:
                for cs in args.cpu_sync:
                    def run(buf, cap, pool):
                        with EnvMode(cap, pool):
                            rx, _, rw, handle, _ = buf.dispatch(
                                x=x, topk_idx=topk, topk_weights=w, num_experts=E, num_max_tokens_per_rank=T,
                                do_cpu_sync=bool(cs), do_expand=bool(expand),
                                allocate_on_comm_stream=False, async_with_compute_stream=False)
                            y = torch.empty_like(rx)
                            y.copy_(rx)   # the "expert": identity
                            out, _, _ = buf.combine(x=y, handle=handle, topk_weights=None if expand else rw,
                                                    allocate_on_comm_stream=False, async_with_compute_stream=False)
                        return rx, handle, out

                    def valid_rows(handle):
                        return (handle.psum_num_recv_tokens_per_expert[-1] if expand
                                else handle.psum_num_recv_tokens_per_scaleup_rank[-1])

                    # correctness and memory, one call each
                    ref, info = None, {}
                    for (cap, pool), buf in bufs.items():
                        dist.barrier(); torch.cuda.synchronize()
                        run(buf, cap, pool)          # warm (JIT, first-call state)
                        torch.cuda.synchronize()
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        rx, handle, out = run(buf, cap, pool)
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated() - base
                        n = int(valid_rows(handle))
                        got = (row_sums(rx, n), out.clone())
                        c = {}
                        if ref is not None:
                            c = dict(recv_equal=bool(torch.equal(ref[0], got[0])), combine_equal=bool(torch.equal(ref[1], got[1])))
                        else:
                            ref = got
                        ok = torch.tensor([float(all(c.values()))], device='cuda')
                        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                        c['all_ranks_equal'] = bool(ok.item())
                        info[(cap, pool)] = dict(c, recv_rows_alloc=rx.shape[0], recv_rows_valid=n,
                                                 recv_x_bytes=rx.numel() * rx.element_size(), peak_tensor_bytes=peak)
                        if cap is not None and cap < -1:
                            # pages each source held on each destination in this call (given back at its end)
                            with EnvMode(cap, pool):
                                have, gotp, pred = buf.runtime.epa_take_state(T).long()
                            held = torch.where(pred > have, have + gotp, pred)
                            allh = [torch.empty_like(held) for _ in range(world)]
                            dist.all_gather(allh, held)
                            per_dst = torch.stack(allh).sum(0)          # pages taken on each receiver
                            row_bytes = rx.shape[1] * rx.element_size() + 128   # hidden + top-k/metadata/padding (14,464 B at H=7,168)
                            info[(cap, pool)].update(pages_taken_max=int(per_dst.max()), pages_taken_mean=per_dst.float().mean().item(),
                                                     page_bytes=(-cap - 1) * row_bytes)
                        del rx, handle, out, got
                    del ref

                    samples = {m: [] for m in bufs}
                    order = list(bufs.items())
                    for rep in range(args.reps):
                        for (cap, pool), buf in order[rep % len(order):] + order[:rep % len(order)]:
                            for _ in range(3):
                                run(buf, cap, pool)
                            torch.cuda.synchronize(); dist.barrier()
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                                t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                h0 = time.perf_counter()
                                t0.record()
                                for _ in range(args.iters):
                                    run(buf, cap, pool)
                                t1.record()
                                torch.cuda.synchronize()
                                h1 = time.perf_counter()
                            kd = {'dispatch': 0.0, 'combine': 0.0}
                            for ev in prof.events():
                                if ev.device_type != torch.autograd.DeviceType.CUDA:
                                    continue
                                if 'dispatch_impl' in ev.name or 'dispatch_copy_epilogue' in ev.name:
                                    kd['dispatch'] += ev.device_time_total
                                elif 'combine_impl' in ev.name or 'combine_reduce_epilogue' in ev.name:
                                    kd['combine'] += ev.device_time_total
                            samples[(cap, pool)].append((kd['dispatch'] / args.iters, kd['combine'] / args.iters,
                                                         t0.elapsed_time(t1) * 1e3 / args.iters,
                                                         (h1 - h0) * 1e6 / args.iters))
                    # decode-style: without the CPU sync the whole step can be a CUDA graph (no host in the loop)
                    graph_us = {m: float('nan') for m in bufs}
                    if not cs:
                        for (cap, pool), buf in bufs.items():
                            try:
                                torch.cuda.synchronize(); dist.barrier()
                                g = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(g):
                                    for _ in range(args.graph_iters):
                                        run(buf, cap, pool)
                                ts = []
                                for _ in range(args.reps + 1):
                                    torch.cuda.synchronize(); dist.barrier()
                                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    t0.record(); g.replay(); t1.record()
                                    torch.cuda.synchronize()
                                    ts.append(t0.elapsed_time(t1) * 1e3 / args.graph_iters)
                                graph_us[(cap, pool)] = st.median(ts[1:])
                                del g
                            except Exception as e:  # noqa: BLE001
                                if rank == 0:
                                    print('graph capture failed:', repr(e)[:200], flush=True)
                                torch.cuda.synchronize()
                    for (cap, pool), buf in bufs.items():
                        v = torch.tensor([st.median(s[i] for s in samples[(cap, pool)]) for i in range(4)]
                                         + [graph_us[(cap, pool)]], device='cuda')
                        allv = [torch.empty_like(v) for _ in range(world)]
                        dist.all_gather(allv, v)
                        vmed = torch.stack(allv).median(0).values
                        # memory: worst rank for the peak and valid rows
                        mem = torch.tensor([info[(cap, pool)]['peak_tensor_bytes'], info[(cap, pool)]['recv_rows_valid']],
                                           device='cuda', dtype=torch.long)
                        allm = [torch.empty_like(mem) for _ in range(world)]
                        dist.all_gather(allm, mem)
                        allm = torch.stack(allm)
                        rec = dict(T=T, routing=routing, expand=expand, cpu_sync=cs,
                                   mode='upstream' if cap is None else f'arena P={-cap - 1}', pages=pool,
                                   buffer_bytes=buf.num_bytes, dispatch_us=vmed[0].item(), combine_us=vmed[1].item(),
                                   step_us=vmed[2].item(), wall_us=vmed[3].item(), graph_step_us=vmed[4].item(),
                                   peak_tensor_bytes_max=int(allm[:, 0].max()),
                                   recv_rows_valid_mean=allm[:, 1].float().mean().item(),
                                   recv_rows_valid_max=int(allm[:, 1].max()),
                                   **{k: v for k, v in info[(cap, pool)].items() if k not in ('peak_tensor_bytes', 'recv_rows_valid')},
                                   max_recv_tokens=int(pairs.sum(0).max()))
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
    p.add_argument('--tokens', default='128')
    p.add_argument('--routing', default='uniform,extreme')
    p.add_argument('--arena', default='64:1.25', help='rows_per_page:beta,...')
    p.add_argument('--expand', default='1')
    p.add_argument('--cpu-sync', default='0,1')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--graph-iters', type=int, default=10)
    p.add_argument('--port', type=int, default=29641)
    p.add_argument('--out', default='')
    a = p.parse_args()
    a.tokens = [int(t) for t in a.tokens.split(',')]
    a.routing = a.routing.split(',')
    a.arena = [tuple(float(v) for v in t.split(':')) for t in a.arena.split(',')] if a.arena else []
    a.expand = [int(t) for t in a.expand.split(',')]
    a.cpu_sync = [int(t) for t in a.cpu_sync.split(',')]
    mp.spawn(worker, args=(8, a), nprocs=8)
