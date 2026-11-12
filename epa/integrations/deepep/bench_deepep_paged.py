"""DeepEP V2.5 expanded layout (do_expand=True), 8 GPUs: upstream vs paged recv_x / y.

Modes, for every tokens-per-rank T and routing:
  upstream, no sync   recv_x / y sized for the worst case R*T*min(k, E_local) rows; CUDA graph
  upstream, CPU sync  host waits for the counts, exact sizes; eager only
  paged               recv_x / y in an epa::Arena of pages of P rows (layout/ep/epa_out.cuh),
                      beta x the mean T*k rows plus one partial page per expert; CUDA graph
  arena + paged       paged outputs and the epa::Arena dispatch receive buffer; CUDA graph
Every mode uses expert_alignment = P, so a page is one grouped-GEMM tile in all of them.

Step = dispatch + expert + combine. The expert is an identity copy recv_x -> y that, like a
grouped GEMM, touches only rows that hold data: upstream rows below the expanded total,
paged rows of pages with an expert (tile_expert >= 0). Both are known only on the GPU.

Memory per GPU (worst rank): EP buffer; recv_x + y as reserved (upstream: the tensors one
call allocates; paged: the two pools) and as used in the call (pages held); PyTorch peak
of one step above what was live before (transient tensors).

usage: python bench_deepep_paged.py --tokens 128,256,512,1024,4096 --routing uniform,extreme
"""
import argparse
import json
import math
import os
import statistics as st
import time
from pathlib import Path

import triton
import triton.language as tl

from bench_deepep import H, E, K, make_routing, torch, dist, mp

ENV = ('EP_EPA_CAP', 'EP_EPA_POOL', 'EP_EPA_ARENA_P', 'EP_EPA_ARENA_PAGES', 'EP_EPA_OUT_P', 'EP_EPA_OUT_PAGES', 'EP_EPA_SHARED_A', 'EP_EPA_SHARED_B', 'EP_EPA_SPILL')


class Env:
    def __init__(self, **kv):
        self.kv = {k: str(v) for k, v in kv.items() if v}

    def __enter__(self):
        for k in ENV:
            os.environ.pop(k, None)
        os.environ.update(self.kv)

    def __exit__(self, *a):
        for k in ENV:
            os.environ.pop(k, None)


@triton.jit
def copy_tiles(src, dst, limit_ptr, list_ptr, list_y_ptr, P: tl.constexpr, HID: tl.constexpr, PAGED: tl.constexpr,
               BLOCK: tl.constexpr):
    """One program per tile of P rows, tiles with data first (as a grouped GEMM's scheduler).
    Upstream: tile t holds data if t * P < limit. Paged: the dense list [count, page...]."""
    t = tl.program_id(0)
    if PAGED:
        valid = t < tl.load(list_ptr)
        page = tl.load(list_ptr + 1 + t, mask=valid, other=0)
        page_y = tl.load(list_y_ptr + 1 + t, mask=valid, other=0)
        lim = 0
    else:
        lim = tl.load(limit_ptr)
        valid = t * P < lim
        page = t
        page_y = t
    if valid:
        for r in range(P):
            if PAGED or t * P + r < lim:   # upstream: stop at the last row (normal layout rows are not P-aligned)
              base = (page * P + r).to(tl.int64) * HID
              base_y = (page_y * P + r).to(tl.int64) * HID
              for c in range(0, HID, BLOCK):
                  offs = c + tl.arange(0, BLOCK)
                  tl.store(dst + base_y + offs, tl.load(src + base + offs))


def row_sums(rx, idx):
    """Order-free fingerprint of rows idx of rx."""
    v = rx[idx].contiguous().view(torch.int32).long()
    w = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=torch.long)
    return torch.sort((v * w).sum(1)).values


def worker(rank, world, args):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    import deep_ep
    group = dist.new_group(list(range(world)))
    L = E // world
    results = []
    for T in args.tokens:
        P = 64 if T >= 4096 else 16                      # page = tile = expert_alignment
        p_hit = 1 - math.comb(E - L, K) / math.comb(E, K)
        if args.expand:
            out_pages = math.ceil(args.beta * T * K / P) + L    # beta x mean expanded rows, one partial page per expert
        else:
            out_pages = math.ceil(args.beta_normal * world * T * p_hit / P) + world   # beta x mean received rows, one partial page per source
        Pd = 64 if T >= 4096 else 16                     # dispatch arena page
        disp_pages = math.ceil(1.25 * world * T * p_hit / Pd) + world
        # Shared pool: one page space for receive, recv_x and y pages; zone A = combine's layout exactly,
        # zone B = the rest, sized beta_shared x the mean pages of all three (+ one partial page per key)
        pb = P * H * 2
        n_c = math.ceil(min(world, K) * T * 14400 / pb)
        keys = L if args.expand else world
        m_x = (T * K if args.expand else world * T * p_hit) / P + keys
        m_recv = world * T * p_hit / (P - 1) + world
        n_total = math.ceil(args.beta_shared * (m_recv + 2 * m_x))
        shared_env = dict(EP_EPA_OUT_P=P, EP_EPA_ARENA_P=P - 1, EP_EPA_SHARED_A=n_c, EP_EPA_SHARED_B=max(1, n_total - n_c))
        if args.spill:
            shared_env['EP_EPA_SPILL'] = 1
        modes = [dict(name='upstream', cs=0, env={}),
                 dict(name='upstream', cs=1, env={}),
                 dict(name='arena+paged', cs=0, env=dict(EP_EPA_OUT_P=P, EP_EPA_OUT_PAGES=out_pages,
                                                         EP_EPA_ARENA_P=Pd, EP_EPA_ARENA_PAGES=disp_pages))]
        if args.shared:
            modes.append(dict(name='shared', cs=0, env=shared_env))
        bufs = {}
        for m in modes:
            key = (m['name'], m['cs'])
            with Env(**m['env']):
                bufs[key] = deep_ep.EPBuffer(group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
                                             allow_hybrid_mode=False, explicitly_destroy=True)
        dummy = torch.zeros(1, dtype=torch.int32, device='cuda')
        NSM = torch.cuda.get_device_properties(rank).multi_processor_count
        for routing in args.routing:
            gen = torch.Generator(device='cuda').manual_seed(1000 * rank + T)
            # --vary N: N independent routings of the same kind, cycled step by step, so the
            # per-destination counts change every call (as in decode)
            routes = []
            for _ in range(max(1, args.vary)):
                tk, ww = make_routing(T, routing, gen)
                routes.append((tk.to(deep_ep.topk_idx_t), ww))
            x = torch.randn((T, H), generator=gen, device='cuda').to(torch.bfloat16)
            step = [0]

            def run(m, buf, fixed=False):
                topk, w = routes[0] if fixed else routes[step[0] % len(routes)]
                step[0] += 1
                paged = 'EP_EPA_OUT_P' in m['env']
                with Env(**m['env']):
                    rx, _, rw, handle, _ = buf.dispatch(
                        x=x, topk_idx=topk, topk_weights=w, num_experts=E, num_max_tokens_per_rank=T,
                        expert_alignment=P, do_cpu_sync=bool(m['cs']), do_expand=bool(args.expand),
                        allocate_on_comm_stream=False, async_with_compute_stream=False)
                    if os.environ.get('BENCH_NO_EXPERT') == '1':
                        y = buf.runtime.epa_out_state()[0] if paged else torch.empty_like(rx)
                    elif paged:
                        y = buf.runtime.epa_out_state()[0]
                        tiles = buf.runtime.epa_out_tiles(L)
                        n_pg = tiles.shape[0] // 2
                        copy_tiles[(rx.shape[0] // P,)](rx, y, dummy, tiles, tiles[n_pg:], P=P, HID=H, PAGED=True, BLOCK=1024)
                    else:
                        y = torch.empty_like(rx)
                        limit = (handle.psum_num_recv_tokens_per_expert[-1:] if args.expand   # expand mode: end of the last expert's rows
                                 else handle.psum_num_recv_tokens_per_scaleup_rank[-1:])
                        copy_tiles[(triton.cdiv(rx.shape[0], P),)](rx, y, limit, dummy, dummy, P=P, HID=H, PAGED=False, BLOCK=1024)
                    out, _, _ = buf.combine(x=y, handle=handle, topk_weights=None,
                                            allocate_on_comm_stream=False, async_with_compute_stream=False)
                return rx, y, handle, out

            # correctness and memory
            ref, info = None, {}
            for m in modes:
                key, buf = (m['name'], m['cs']), bufs[(m['name'], m['cs'])]
                dist.barrier(); torch.cuda.synchronize()
                run(m, buf)
                torch.cuda.synchronize()
                if os.environ.get('SPILL_DEBUG') and 'EP_EPA_OUT_P' in m['env']:
                    print('CALL1', rank, m['name'], buf.runtime.epa_out_state()[4].tolist(), flush=True)
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                rx, y, handle, out = run(m, buf, fixed=True)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                n_tok = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1])   # rows of metadata that hold data
                if args.expand or 'EP_EPA_OUT_P' in m['env']:
                    slots = handle.recv_src_metadata[:n_tok, 2:].flatten()   # expanded rows, or the paged normal layout's page rows
                    slots = slots[slots >= 0].long()
                else:
                    slots = torch.arange(n_tok, device='cuda')
                got = (row_sums(rx, slots), out.clone())
                c = {}
                if ref is None:
                    ref = got
                else:
                    c = dict(recv_equal=bool(torch.equal(ref[0], got[0])), combine_equal=bool(torch.equal(ref[1], got[1])))
                ok = torch.tensor([float(all(c.values()))], device='cuda')
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                c['all_ranks_equal'] = bool(ok.item())
                xy_bytes = 2 * rx.numel() * rx.element_size()          # recv_x and y as reserved
                used_rows = int(slots.numel())
                held = 0
                if 'EP_EPA_OUT_P' in m['env']:
                    stats = buf.runtime.epa_out_state()[4].tolist()     # held, failed, dropped, max held
                    held, c['out_failed_takes'], c['out_dropped_rows'] = stats[0], stats[1], stats[2]
                    c['out_borrowed_pages'] = stats[11]
                    if os.environ.get('SPILL_DEBUG'):
                        print('RANKSTAT', rank, T, routing, m['name'], stats, flush=True)
                    xy_bytes += 0   # pools are persistent: counted as reserved
                recv_held = 0
                if 'EP_EPA_ARENA_P' in m['env']:
                    with Env(**m['env']):
                        recv_held = int(buf.runtime.epa_recv_held(T).sum())
                info[key] = dict(c, xy_bytes=xy_bytes, peak=peak, used_rows=used_rows, held_pages=held, recv_held=recv_held,
                                 rows_alloc=rx.shape[0])
                del rx, y, handle, out, got
            del ref

            # timing: eager for all, CUDA graph without the CPU sync
            samples = {k: [] for k in bufs}
            for rep in range(args.reps):
                order = modes[rep % len(modes):] + modes[:rep % len(modes)]
                for m in order:
                    key, buf = (m['name'], m['cs']), bufs[(m['name'], m['cs'])]
                    for _ in range(3):
                        run(m, buf)
                    torch.cuda.synchronize(); dist.barrier()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(args.iters):
                        run(m, buf)
                    t1.record()
                    torch.cuda.synchronize()
                    samples[key].append(t0.elapsed_time(t1) * 1e3 / args.iters)
            if args.profile:
                for m in modes:
                    key, buf = (m['name'], m['cs']), bufs[(m['name'], m['cs'])]
                    torch.cuda.synchronize(); dist.barrier()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(args.iters):
                            run(m, buf)
                        torch.cuda.synchronize()
                    agg = {}
                    for ev in prof.events():
                        if ev.device_type == torch.autograd.DeviceType.CUDA:
                            nm = ev.name.split('<')[0][:40]
                            agg[nm] = agg.get(nm, 0.0) + ev.device_time_total / args.iters
                    if rank == 0:
                        print('PROFILE', T, routing, key, {k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:8]}, flush=True)
            graph_us = {k: float('nan') for k in bufs}
            for m in modes:
                key, buf = (m['name'], m['cs']), bufs[(m['name'], m['cs'])]
                if m['cs']:
                    continue
                torch.cuda.synchronize(); dist.barrier()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(args.graph_iters):
                        run(m, buf)
                ts = []
                for _ in range(args.reps + 1):
                    torch.cuda.synchronize(); dist.barrier()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record(); g.replay(); t1.record()
                    torch.cuda.synchronize()
                    ts.append(t0.elapsed_time(t1) * 1e3 / args.graph_iters)
                graph_us[key] = st.median(ts[1:])
                if args.profile:
                    torch.cuda.synchronize(); dist.barrier()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        g.replay()
                        torch.cuda.synchronize()
                    agg = {}
                    for ev in prof.events():
                        if ev.device_type == torch.autograd.DeviceType.CUDA:
                            nm = ev.name.split('<')[0].replace('void deep_ep::ep::', '')[:28]
                            agg[nm] = agg.get(nm, 0.0) + ev.device_time_total / args.graph_iters
                    v = torch.tensor([agg.get(k, 0.0) for k in sorted(agg)], device='cuda')
                    allv = [torch.empty_like(v) for _ in range(world)]
                    dist.all_gather(allv, v)
                    med = torch.stack(allv).median(0).values.tolist()
                    if rank == 0:
                        print('GPROFILE', T, routing, key, {k: round(x, 1) for k, x in zip(sorted(agg), med)}, flush=True)
                        print('GPROFILE_MAX', T, routing, key, {k: round(x, 1) for k, x in zip(sorted(agg), torch.stack(allv).max(0).values.tolist())}, flush=True)
                        print('GPROFILE_RANKS', T, routing, key, [round(sum(r.tolist()), 1) for r in allv], flush=True)
                del g
            for m in modes:
                key, buf = (m['name'], m['cs']), bufs[(m['name'], m['cs'])]
                i = info[key]
                v = torch.tensor([st.median(samples[key]), graph_us[key], i['xy_bytes'], i['peak'], i['used_rows'],
                                  i['held_pages'], i['recv_held'], i.get('out_borrowed_pages', 0),
                                  i.get('out_failed_takes', 0), i.get('out_dropped_rows', 0)], device='cuda', dtype=torch.float64)
                allv = [torch.empty_like(v) for _ in range(world)]
                dist.all_gather(allv, v)
                allv = torch.stack(allv)
                pool_bytes = 2 * out_pages * P * H * 2 if 'EP_EPA_OUT_P' in m['env'] else 0
                rec = dict(T=T, routing=routing, vary=args.vary, expand=args.expand, mode=m['name'], cpu_sync=m['cs'], P=P, out_pages=out_pages,
                           beta=args.beta, buffer_bytes=buf.num_bytes,
                           eager_us=allv[:, 0].median().item(), graph_us=allv[:, 1].median().item(),
                           xy_reserved_bytes=0 if m['name'] == 'shared' else (int(allv[:, 2].max()) if not pool_bytes else pool_bytes),   # shared: inside buffer_bytes
                           xy_used_bytes=int(allv[:, 4].max()) * 2 * H * 2,
                           pages_held_max=int(allv[:, 5].max()),
                           recv_pages_held_max=int(allv[:, 6].max()),
                           xy_pages_bytes_max=int(allv[:, 5].max()) * P * H * 2 * 2,
                           peak_tensor_bytes_max=int(allv[:, 3].max()),
                           used_rows_max=int(allv[:, 4].max()), used_rows_mean=allv[:, 4].mean().item(),
                           rows_alloc=i['rows_alloc'],
                           borrowed_pages_max=int(allv[:, 7].max()), failed_takes_all=int(allv[:, 8].sum()),
                           dropped_rows_all=int(allv[:, 9].sum()),
                           **{k: v for k, v in i.items() if k in ('recv_equal', 'combine_equal', 'all_ranks_equal',
                                                                  'out_failed_takes', 'out_dropped_rows')})
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
    p.add_argument('--beta', type=float, default=2.0, help='paged pool: beta x the mean T*k expanded rows')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--graph-iters', type=int, default=10)
    p.add_argument('--port', type=int, default=29671)
    p.add_argument('--out', default='')
    p.add_argument('--profile', type=int, default=0)
    p.add_argument('--expand', type=int, default=1, help='1: expanded layout, 0: normal layout')
    p.add_argument('--beta-normal', type=float, default=1.25, help='normal layout pool: beta x the mean received rows')
    p.add_argument('--shared', type=int, default=0, help='also run the shared pool (receive, recv_x and y pages in the EP buffer)')
    p.add_argument('--spill', type=int, default=0, help='shared pool: rows that do not fit borrow peer GPUs\' free pages (EP_EPA_SPILL)')
    p.add_argument('--beta-shared', type=float, default=1.25, help='shared pool: beta x the mean pages of receive + recv_x + y')
    p.add_argument('--world', type=int, default=8)
    p.add_argument('--vary', type=int, default=1, help='routings cycled per step (1: the same routing every step)')
    a = p.parse_args()
    a.tokens = [int(t) for t in a.tokens.split(',')]
    a.routing = a.routing.split(',')
    mp.spawn(worker, args=(a.world, a), nprocs=a.world)
