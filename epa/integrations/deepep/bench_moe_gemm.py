"""One MoE layer on DeepEP V2.5 + DeepGEMM: upstream vs an EPArena segment.

usage: python bench_moe_gemm.py --world 4 --tokens 512,1024,2048,4096 --routing uniform

Layer (captured in a CUDA graph, no CPU sync anywhere):
  dispatch (expanded layout, expert_alignment 128) -> GEMM 1 (DeepGEMM m-grouped BF16, psum
  layout: x [M, 7168] x W13 [L, 4096, 7168]) -> SiLU(gate) * up (rows < psum[-1] only) ->
  GEMM 2 (a [M, 2048] x W2 [L, 7168, 2048]) -> combine
Modes:
  upstream  DeepEP allocates recv_x for the worst case (R * T * min(k, L) rows + alignment);
            h, a and y have the same rows
  segment   EP_EPA_SEG_ROWS = beta x the mean expanded rows (+ alignment of every expert):
            recv_x, h and a have that many rows, and y is leased in place: GEMM 2 writes into
            recv_x's rows, which GEMM 1 has finished reading; combine writes its output into a,
            which GEMM 2 has finished reading. Rows past the segment are borrowed from other GPUs'
            unused segment tails (no separate spill buffer)
Checks: the combined output is bitwise the same in both modes, every rank's rows fit the
segment, and a few tokens match a torch reference of the layer.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from bench_deepep import E, K, make_routing, dist, mp

H, I, ALIGN = 7168, 2048, 128
# KV cache per token for --kv-*: DeepSeek-V3 MLA, (512 latent + 64 rope) BF16 values x 61 layers
KV_TOKEN_BYTES = (512 + 64) * 2 * 61
# overflow: rows past the segment are borrowed from peers' unused segment tails and run through
# WORK buffers in NP pieces (pull, GEMMs; their y is leased into the segment's dead h)


@triton.jit
def seg_rows_copy(work, hdr, base_row, total_ptr, H: tl.constexpr, BLOCK: tl.constexpr, PUSH: tl.constexpr):
    """Row j of a piece <-> logical row base_row + j of the segment (>= C: a borrowed row on another
    GPU, found in the header's chunk table). PUSH = 0: pull into work; 1: push work back."""
    j = tl.program_id(0)
    r = base_row + j
    if r >= tl.load(total_ptr):
        return
    n = tl.load(hdr + 4)
    addr = tl.zeros((), dtype=tl.int64)
    found = 0
    for k in range(16):
        first = tl.load(hdr + 16 + 4 * k + 1)
        rows = tl.load(hdr + 16 + 4 * k + 2)
        hit = (k < n) & (found == 0) & (r >= first) & (r < first + rows)
        lender = tl.load(hdr + 16 + 4 * k)
        phys = tl.load(hdr + 16 + 4 * k + 3)
        lo = tl.load(hdr + 128 + 2 * lender).to(tl.uint32).to(tl.int64)
        hi = tl.load(hdr + 128 + 2 * lender + 1).to(tl.int64)
        base = (hi << 32) | lo
        addr = tl.where(hit, base + (phys + r - first).to(tl.int64) * (H * 2), addr)
        found = tl.where(hit, 1, found)
    if found == 0:
        return
    ptr = addr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    for c0 in range(0, H, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        if PUSH:
            tl.store(ptr + c, tl.load(work + j.to(tl.int64) * H + c))
        else:
            tl.store(work + j.to(tl.int64) * H + c, tl.load(ptr + c))


@triton.jit
def silu_mul(h, a, total_ptr, I: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    total = tl.load(total_ptr)
    if r >= total:
        return
    for c0 in range(0, I, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        g = tl.load(h + r.to(tl.int64) * (2 * I) + c).to(tl.float32)
        u = tl.load(h + r.to(tl.int64) * (2 * I) + I + c).to(tl.float32)
        tl.store(a + r.to(tl.int64) * I + c, (g / (1 + tl.exp(-g)) * u).to(tl.bfloat16))


@triton.jit
def seg_redirect(hdr, R: tl.constexpr, cap_rows: tl.constexpr):
    """After the borrowed pieces: their y lives in the leased GEMM 1 output buffer (chunk lender R),
    so combine reads it locally."""
    C = tl.load(hdr + 0)
    total = tl.load(hdr + 5)
    nb = tl.minimum(tl.maximum(total - C, 0), cap_rows)
    o = hdr + 16
    tl.store(o, R)
    tl.store(o + 1, C)
    tl.store(o + 2, nb)
    tl.store(o + 3, 0)
    tl.store(hdr + 4, tl.where(nb > 0, 1, 0))


def expert_weights(e, device):
    g = torch.Generator(device=device).manual_seed(10_000 + e)
    w13 = (torch.randn(2 * I, H, generator=g, device=device) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(H, I, generator=g, device=device) * 0.02).to(torch.bfloat16)
    return w13, w2


def worker(rank, world, args):
    try:
        _worker(rank, world, args)
    except Exception as ex:   # out of memory anywhere (graph capture, DeepEP, NCCL): report and end the job
        if isinstance(ex, torch.OutOfMemoryError) or 'out of memory' in str(ex).lower():
            print(f'OOM rank {rank} mode {",".join(args.modes)}: {str(ex).splitlines()[0]}', flush=True)
            os._exit(3)
        raise


def _worker(rank, world, args):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    import deep_ep
    import deep_gemm
    group = dist.new_group(list(range(world)))
    L = E // world
    dev = torch.device('cuda', rank)
    # NL layers that cycle (layer 0's experts use the reference seeds)
    NL = args.layers
    W13s = [torch.empty(L, 2 * I, H, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    W2s = [torch.empty(L, H, I, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    for li in range(NL):
        for le in range(L):
            W13s[li][le], W2s[li][le] = expert_weights(li * 100000 + rank * L + le, dev)
    torch.cuda.synchronize()
    W13E, W2E = L * 2 * I * H, L * H * I
    # EPArena host tier (include/epa/host_tier.hpp, python/epa_host_torch.py), used only when every
    # GPU is out of reserved memory: the GEMM 1 weights of all layers become epa PagedWeights. Each
    # layer is one contiguous tensor for DeepGEMM, but the last --host-mb of every layer maps to one
    # shared resident tail; all tails wait in pinned host memory. Right after a layer's last GEMM 1,
    # a side stream pages the next layer's tail in (--page-mb a copy); the next GEMM 1 waits for it.
    # HBM saved per GPU: (NL - 1) x tail.
    host = dict(state=0, pending=[False] * NL)
    EV = 0
    if args.host_mb > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
        from epa_host_torch import paged_weights
        pw, wv = paged_weights(W13s, int(args.host_mb * 2**20))
        EV = pw.tail_bytes() // 2                      # elements (2 MB granularity)
        PAGE = int(args.page_mb * 2**20)
        s3 = torch.cuda.Stream()
        ev_r_free, ev_paged = torch.cuda.Event(), [torch.cuda.Event() for _ in range(NL)]
        with torch.cuda.stream(s3):
            pw.page_in(0, PAGE)
        torch.cuda.synchronize()

    def ensure_r(li):   # outside timing: make the shared tail hold layer li's
        if EV and host['state'] != li:
            pw.page_in(li, PAGE); torch.cuda.synchronize()
            host['state'] = li
        host['pending'] = [False] * NL

    def page_in(li):   # every GEMM 1 of layer li is done: page the next layer's tail in (side stream)
        nxt = (li + 1) % NL
        ev_r_free.record()
        s3.wait_event(ev_r_free)
        with torch.cuda.stream(s3):
            pw.page_in(nxt, PAGE)
            ev_paged[nxt].record(s3)
        host['state'], host['pending'][nxt] = nxt, True

    def weights(m, li):
        if EV and m['seg']:
            return wv[li], W2s[li]
        return W13s[li], W2s[li]

    for T in args.tokens:
        # segment: beta x the expected expanded rows per GPU (each expert rounds up to 128 rows,
        # half a block on average; at small T every expert takes at least one block)
        mean_rows = max(T * K + L * ALIGN // 2, L * ALIGN)
        seg_rows = math.ceil(args.beta * mean_rows)
        seg_rows = (seg_rows + ALIGN - 1) // ALIGN * ALIGN
        # only the segment mode needs the segment and the borrowing buffers (a run with
        # --modes upstream or cpusync allocates none of them, so a --budget-mb test is fair)
        SEG = 'segment' in args.modes
        seg_bytes = 4096 + seg_rows * H * 2
        if SEG:
            os.environ['EP_EPA_SEG_BYTES'] = str(seg_bytes)
        buf = deep_ep.EPBuffer(group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
                               allow_hybrid_mode=False, explicitly_destroy=True)
        os.environ.pop('EP_EPA_SEG_BYTES', None)
        if SEG:
            buf.runtime.epa_seg_init(seg_rows, H * 2)
            dist.barrier()
            hdr = buf.runtime.epa_seg_header()
            WORK, NP, sr = args.work, args.pieces, seg_rows
        else:
            hdr = torch.zeros(1024, dtype=torch.int32, device='cuda')
            WORK, NP, sr = 0, 0, 1
        # segment mode keeps h / a persistent: after the local GEMMs h is IDLE and is leased to the
        # borrowed rows' y (lender R in the chunk table); pieces are pulled into 2 work buffers
        h_seg = torch.empty(sr, 2 * I, device='cuda', dtype=torch.bfloat16)
        a_seg = torch.empty(sr, I, device='cuda', dtype=torch.bfloat16)
        y_cap = sr * 2 * I // H                          # borrowed y rows the leased h can hold
        y_store = h_seg.view(-1)[: y_cap * H].view(y_cap, H)
        if WORK > y_cap:   # small T: a piece's y must fit the leased h
            WORK = y_cap // ALIGN * ALIGN
        if WORK and NP == 0:   # --pieces 0: as many pieces as the leased h can hold y for (empty ones are
            NP = -(-y_cap // WORK)   # skipped on the GPU with --cond), so every storable borrowed row is covered
        # piece p covers rows [p * WORK, p * WORK + caps[p]) of the borrowed rows; its y goes to the same
        # rows of the leased h, so the last piece is shorter where the leased h ends
        caps = [min(WORK, (y_cap - p * WORK) // ALIGN * ALIGN) for p in range(NP)] if WORK else []
        caps = [c for c in caps if c > 0]
        NP, COVER = len(caps), sum(caps)
        # --pull gemm: each piece's GEMM 1 pulls its rows (epa::RowPull), and earlier GEMMs prefetch
        # them on their spare warps: local GEMM 1 -> piece 0, local GEMM 2 -> piece 1, piece p's
        # GEMM 1 -> piece p + 1 (2 work buffers). gemm-nopf: no prefetch, 1 work buffer.
        # kernel: a copy kernel on a side stream, 2 work buffers.
        PULL_GEMM = args.pull in ('gemm', 'gemm-nopf')
        PF = args.pull == 'gemm'
        works = [torch.empty(max(WORK, 1), H, device='cuda', dtype=torch.bfloat16)
                 for _ in range(2 if NP > 1 and args.pull != 'gemm-nopf' else 1)]
        stamps = [torch.zeros(max(WORK, 1), dtype=torch.int64, device='cuda') for _ in works]   # >= rows / 2
        def piece_pull(p):   # (stage, header, first logical row, stamps) of piece p, for epa_prefetch
            return (works[p % len(works)][:caps[p]], hdr, seg_rows + p * WORK, stamps[p % len(works)])
        h_p = torch.empty(max(WORK, 1), 2 * I, device='cuda', dtype=torch.bfloat16)
        a_p = torch.empty(max(WORK, 1), I, device='cuda', dtype=torch.bfloat16)
        s2 = torch.cuda.Stream()
        ev_d = torch.cuda.Event()
        ev_pull = [torch.cuda.Event() for _ in range(max(NP, 1))]
        ev_g1 = [torch.cuda.Event() for _ in range(max(NP, 1))]
        hdr[128 + 2 * world: 130 + 2 * world] = torch.tensor([y_store.data_ptr()], dtype=torch.int64).view(torch.int32).cuda()
        # combine's output, leased into the segment's a (dead once the local GEMM 2 has read it)
        out_seg = a_seg.view(-1)[: T * H].view(T, H) if a_seg.numel() >= T * H else None
        torch.cuda.synchronize(); dist.barrier()
        modes = [dict(name='upstream', seg=0), dict(name='cpusync', seg=0, cs=1), dict(name='segment', seg=seg_rows)]
        modes = [m for m in modes if m['name'] in args.modes]
        setup_bytes = ((seg_bytes + h_seg.numel() * 2 + a_seg.numel() * 2
                        + (len(works) * WORK * H + WORK * 3 * I) * 2) if SEG else 0)
        ballast = None
        if args.budget_mb > 0:
            # memory budget: everything this layer holds above the weights and DeepEP's own buffer
            # (the same quantity as peak_MB) must fit in --budget-mb, plus --slack-mb for the CUDA
            # runtime (DeepGEMM / Triton / NCCL workspaces). A ballast tensor takes the rest of the
            # GPU, so going past the budget is a real CUDA out-of-memory error.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            keep = int((args.budget_mb + args.slack_mb) * 1e6) - setup_bytes
            ballast = torch.empty(max(free - keep, 0), dtype=torch.uint8, device='cuda')
            torch.cuda.synchronize()
            print(f'BUDGET rank {rank}: {args.budget_mb:.0f} MB budget, {setup_bytes / 1e6:.0f} MB of it set up '
                  f'(segment and borrowing buffers), {torch.cuda.mem_get_info()[0] / 1e6:.0f} MB left free', flush=True)
            dist.barrier()

        def kv_pool(keep):   # serving-engine fill: a KV-cache pool takes all free memory but `keep` bytes
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            pool = torch.empty(max(free - keep, 0), dtype=torch.uint8, device='cuda')
            torch.cuda.synchronize()
            print(f'KV rank {rank}: pool {pool.numel() / 1e6:.0f} MB = {pool.numel() // KV_TOKEN_BYTES} tokens, '
                  f'reserve {(keep + setup_bytes) / 1e6:.0f} MB for the layer ({setup_bytes / 1e6:.0f} MB of it set up), '
                  f'{torch.cuda.mem_get_info()[0] / 1e6:.0f} MB left free', flush=True)
            dist.barrier()
            return pool
        reserve_b = 0
        if args.kv_reserve_mb > 0:   # a fixed reserve (MB, counted like peak_MB: segment buffers included)
            reserve_b = int(args.kv_reserve_mb * 1e6)
            ballast = kv_pool(reserve_b - setup_bytes)
        routings = ([args.kv_profile] if args.kv_profile else []) + args.routing
        if args.kv_profile:   # profile pass, like vLLM / SGLang: measure what a full pass takes, no pool yet
            torch.cuda.empty_cache(); torch.cuda.synchronize(); dist.barrier()
            free0, res0 = torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        for ri, routing in enumerate(routings):
            profiling = bool(args.kv_profile) and ri == 0
            if args.kv_profile and ri == 1:   # the pool: everything but what the profile pass took + margin
                torch.cuda.synchronize()
                res1 = torch.cuda.memory_reserved()
                need = (torch.cuda.max_memory_reserved() - res0) + max(0, (free0 - torch.cuda.mem_get_info()[0]) - (res1 - res0))
                keep = need + int(args.kv_margin_mb * 1e6)
                print(f'PROFILE_NEED rank {rank}: {need / 1e6:.0f} MB on {routings[0]} routing '
                      f'(torch reserved peak {(torch.cuda.max_memory_reserved() - res0) / 1e6:.0f} MB)', flush=True)
                reserve_b = keep + setup_bytes
                ballast = kv_pool(keep)
            gen = torch.Generator(device='cuda').manual_seed(1000 * rank + T)
            topk, w = make_routing(T, routing, gen)
            topk = topk.to(deep_ep.topk_idx_t)
            x = (torch.randn((T, H), generator=gen, device='cuda') * 0.5).to(torch.bfloat16)

            def layer(m):
                li = m.get('li', 0)
                m['li'] = (li + 1) % NL
                W13, W2 = weights(m, li)
                paging = bool(EV and m['seg'])
                if paging and host['pending'][li]:   # GEMM 1 needs this layer's part of R
                    torch.cuda.current_stream().wait_event(ev_paged[li])
                    host['pending'][li] = False
                if m['seg']:
                    os.environ['EP_EPA_SEG_ROWS'] = str(m['seg'])
                else:
                    os.environ.pop('EP_EPA_SEG_ROWS', None)
                rx, _, _, handle, _ = buf.dispatch(
                    x=x, topk_idx=topk, topk_weights=w, num_experts=E, num_max_tokens_per_rank=T,
                    expert_alignment=ALIGN, do_cpu_sync=bool(m.get('cs')), do_expand=True,
                    allocate_on_comm_stream=False, async_with_compute_stream=False)
                os.environ.pop('EP_EPA_SEG_ROWS', None)
                psum = handle.psum_num_recv_tokens_per_expert
                M = rx.shape[0]
                assert not m['seg'] or M == seg_rows   # piece_pull's first logical row is C + p * WORK
                if m['seg'] and m.get('snap'):   # the chunk table as dispatch wrote it (eager runs only)
                    m['chunks'] = hdr[:96].clone()
                if m['seg']:   # local rows: the segment's first C rows
                    total = psum[-1:]
                    psum_l = psum.clamp(max=M)
                    if WORK and NP and not PULL_GEMM:   # borrowed pieces: pulls start now on s2, overlapping the local GEMMs
                        ev_d.record()
                        s2.wait_event(ev_d)
                        with torch.cuda.stream(s2):   # the first pieces now; later ones once their buffer is free
                            for p in range(min(NP, len(works))):
                                seg_rows_copy[(WORK,)](works[p], hdr, M + p * WORK, total, H=H, BLOCK=1024, PUSH=0)
                                ev_pull[p].record(s2)
                    h, a = h_seg, a_seg
                else:
                    psum_l = psum
                    h = torch.empty(M, 2 * I, device='cuda', dtype=torch.bfloat16)
                    a = torch.empty(M, I, device='cuda', dtype=torch.bfloat16)
                pf = PF and m['seg'] and WORK and NP
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(rx, W13, h, psum_l, use_psum_layout=True,
                                                            epa_prefetch=piece_pull(0) if pf else None)
                pieces_on = bool(m['seg'] and WORK and NP)
                if paging and not pieces_on:
                    page_in(li)
                silu_mul[(M,)](h, a, psum_l[-1:], I=I, BLOCK=1024)
                del h
                y = rx if m['seg'] else torch.empty_like(rx)   # segment: y leased in recv_x's rows
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, W2, y, psum_l, use_psum_layout=True,
                                                            epa_prefetch=piece_pull(1) if pf and NP > 1 else None)
                del a
                if m['seg']:
                    if WORK and NP:   # every piece's row counts in one go: row p = (psum - base_p) clamped to [0, WORK]
                        if m.get('bases_M') != M:
                            m['bases'] = (M + WORK * torch.arange(NP, device='cuda', dtype=torch.int32)).view(NP, 1)
                            m['caps'] = torch.tensor(caps, device='cuda', dtype=torch.int32).view(NP, 1)
                            m['bases_M'] = M
                        psum_all = torch.minimum((psum.view(1, -1) - m['bases']).clamp_(min=0), m['caps']).to(torch.int32)
                    # --cond (CUDA graph only): piece p runs inside a conditional IF node "piece p has rows",
                    # nested in piece p - 1's, so the GPU skips every empty piece without launching it
                    gcap = m.get('graph') if (args.cond and PULL_GEMM) else None
                    opened = 0
                    for p in range(NP if WORK else 0):   # borrowed rows: GEMMs on the pulled piece, y into the leased h
                        psum_p = psum_all[p]
                        if gcap is not None and (args.cond == 1 or p == 0):   # 1: chain; 2: one IF around all
                            gcap.begin_capture_to_if_node(psum_all[p, -1:] > 0)
                            opened += 1
                        if PULL_GEMM:   # GEMM 1 pulls the piece's rows into the work buffer itself
                            stage, _, base, st_ = piece_pull(p)
                            deep_gemm.m_grouped_bf16_gemm_nt_contiguous_epa_pull(
                                stage, W13, h_p[:caps[p]], psum_p, hdr, base, st_,
                                epa_prefetch=piece_pull(p + 1) if PF and args.pf_from == 'g1' and 1 <= p < NP - 1 else None)
                        else:
                            torch.cuda.current_stream().wait_event(ev_pull[p])
                            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(works[p % len(works)][:caps[p]], W13, h_p[:caps[p]], psum_p, use_psum_layout=True)
                        if not PULL_GEMM:
                            ev_g1[p].record()
                        if not PULL_GEMM and p + len(works) < NP:   # its work buffer is free: pull a later piece into it
                            q = p + len(works)
                            s2.wait_event(ev_g1[p])
                            with torch.cuda.stream(s2):
                                seg_rows_copy[(WORK,)](works[q % len(works)], hdr, M + q * WORK, total, H=H, BLOCK=1024, PUSH=0)
                                ev_pull[q].record(s2)
                        silu_mul[(WORK,)](h_p, a_p, psum_p[-1:], I=I, BLOCK=1024)
                        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(   # g2: its buffer is free, prefetch piece p + 2
                            a_p[:caps[p]], W2, y_store[p * WORK:p * WORK + caps[p]], psum_p, use_psum_layout=True,
                            epa_prefetch=piece_pull(p + 2) if PF and args.pf_from == 'g2' and p + 2 < NP else None)
                    for _ in range(opened):
                        gcap.end_capture_to_conditional_node()
                    if WORK and NP:   # combine reads the borrowed rows' y from the leased h, locally
                        seg_redirect[(1,)](hdr, R=world, cap_rows=COVER)
                if paging and pieces_on:   # the borrowed pieces' GEMM 1 came last
                    page_in(li)
                out, _, _ = buf.combine(x=y, handle=handle, topk_weights=None,
                                        allocate_on_comm_stream=False, async_with_compute_stream=False,
                                        out=out_seg if m['seg'] else None)
                return out, handle, M

            # correctness and memory (per layer: everything allocated above the weights)
            outs, info = {}, {}
            for m in modes:
                try:
                    m['li'] = NL - 1; ensure_r(NL - 1)
                    layer(m)          # the last layer (warm-up), then layer 0 measured
                    torch.cuda.synchronize()
                    ensure_r(0)
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    m['snap'] = True
                    out, handle, M = layer(m)
                    m['snap'] = False
                    torch.cuda.synchronize()
                except torch.OutOfMemoryError as ex:   # past the budget: report and end the job
                    print(f'OOM rank {rank} mode {m["name"]}: {str(ex).splitlines()[0]}', flush=True)
                    os._exit(3)
                peak = torch.cuda.max_memory_allocated() - base
                total = int(handle.psum_num_recv_tokens_per_expert[-1])
                outs[m['name']] = out.clone()
                h_ = hdr.tolist() if m['seg'] else [0] * 8
                if m['seg']:   # where this GPU's rows went: segment, other GPUs
                    ch = m['chunks'].tolist()
                    pieces = [(ch[16 + 4 * k], ch[16 + 4 * k + 2]) for k in range(ch[4])]
                    print('ROWS ' + json.dumps(dict(rank=rank, rows=total, segment=min(total, M),
                                                    borrowed={str(g): r for g, r in pieces if g < world})), flush=True)
                info[m['name']] = dict(rows_alloc=M, rows_used=total,
                                       peak=peak + (setup_bytes if m['seg'] else 0),
                                       borrowed=max(0, total - M) - (h_[6] if m['seg'] else 0), not_placed=h_[6] if m['seg'] else 0,
                                       beyond_piece=max(0, total - M - COVER) if m['seg'] else 0)
                del out, handle
            if args.save_out and not profiling:   # this rank's combined output, to compare runs made with different --modes
                torch.save(outs[modes[0]['name']].cpu(), f'{args.save_out}.T{T}.{routing}.r{rank}.pt')
            ref_out = (torch.load(f'{args.compare_to}.T{T}.{routing}.r{rank}.pt').cuda() if args.compare_to and not profiling
                       else outs[modes[0]['name']])
            eq = torch.tensor([float(all(torch.equal(ref_out, o) for o in outs.values()))], device='cuda')
            fits = torch.tensor([float(info['segment']['not_placed'] == 0 and info['segment']['beyond_piece'] == 0) if SEG else 1.0], device='cuda')
            dist.all_reduce(eq, op=dist.ReduceOp.MIN)
            dist.all_reduce(fits, op=dist.ReduceOp.MIN)
            # reference for 2 tokens: sum over the token's top-k experts of W2 act(W13 x)
            ref_err = 0.0
            for t in (range(2) if ballast is None and not args.kv_profile else []):   # (needs ~0.3 GB; skipped under a budget)
                acc = torch.zeros(H, device='cuda', dtype=torch.float32)
                for e in topk[t].tolist():
                    w13, w2 = expert_weights(e, dev)
                    hh = w13.float() @ x[t].float()
                    aa = (torch.nn.functional.silu(hh[:I]) * hh[I:]).to(torch.bfloat16).float()
                    acc += (w2.float() @ aa).to(torch.bfloat16).float()
                got = outs[modes[-1]['name']][t].float()
                ref_err = max(ref_err, ((got - acc).norm() / acc.norm()).item())
                del w13, w2
            err = torch.tensor([ref_err], device='cuda')
            dist.all_reduce(err, op=dist.ReduceOp.MAX)
            del outs

            if args.profile:   # per-kernel time on the GPU with the most rows (eager)
                rows_t = torch.tensor([float(info['upstream']['rows_used'])], device='cuda')
                allr = [torch.empty_like(rows_t) for _ in range(world)]
                dist.all_gather(allr, rows_t)
                busiest = int(torch.cat(allr).argmax())
                for m in modes:
                    for _ in range(2): layer(m)
                    torch.cuda.synchronize(); dist.barrier()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(5): layer(m)
                        torch.cuda.synchronize()
                    agg = {}
                    for ev in prof.events():
                        if ev.device_type == torch.autograd.DeviceType.CUDA:
                            nm = ev.name.split('<')[0].replace('void deep_ep::ep::', '').replace('deep_gemm::', '')[:30]
                            agg[nm] = agg.get(nm, 0.0) + ev.device_time_total / 5
                    if rank == busiest:
                        print('PROFILE', T, routing, m['name'], 'rank', rank,
                              {k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}, 'total', round(sum(agg.values()), 1), flush=True)
                    dist.barrier()

            # timing: CUDA graph of graph_iters layers, replayed; median over ranks
            graph_us = {}
            for m in modes:
                torch.cuda.synchronize(); dist.barrier()
                for _ in range(2):
                    layer(m)
                torch.cuda.synchronize()
                times = []
                if m.get('cs'):   # the CPU sync rules out CUDA graphs: eager, back to back
                    m.pop('graph', None)
                    for _ in range(args.reps):
                        torch.cuda.synchronize(); dist.barrier()
                        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        t0.record()
                        for _ in range(args.graph_iters):
                            layer(m)
                        t1.record()
                        torch.cuda.synchronize()
                        times.append(t0.elapsed_time(t1) * 1e3 / args.graph_iters)
                else:
                    torch.cuda.synchronize()
                    m['li'] = 0; ensure_r(0)   # the graph starts at layer 0 with no copy pending from outside it
                    g = torch.cuda.CUDAGraph()
                    m['graph'] = g   # for --cond: conditional nodes are added to this graph
                    with torch.cuda.graph(g):
                        for _ in range(args.graph_iters):
                            layer(m)
                        if EV and m['seg']:   # join the last page-in so the replay ends with R ready for layer 0
                            torch.cuda.current_stream().wait_event(ev_paged[host['state']])
                            host['pending'] = [False, False]
                    m.pop('graph', None)
                    for _ in range(args.reps):
                        torch.cuda.synchronize(); dist.barrier()
                        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        t0.record(); g.replay(); t1.record()
                        torch.cuda.synchronize()
                        times.append(t0.elapsed_time(t1) * 1e3 / args.graph_iters)
                    del g
                v = torch.tensor([st.median(times)], device='cuda', dtype=torch.float64)
                allv = [torch.empty_like(v) for _ in range(world)]
                dist.all_gather(allv, v)
                graph_us[m['name']] = dict(median=torch.cat(allv).median().item(), max=torch.cat(allv).max().item())
            for m in modes:
                v = torch.tensor([info[m['name']]['peak'], info[m['name']]['rows_used'], info[m['name']]['rows_alloc'],
                                  info[m['name']].get('borrowed', 0), ballast.numel() if ballast is not None else 0, reserve_b],
                                 device='cuda', dtype=torch.float64)
                allv = [torch.empty_like(v) for _ in range(world)]
                dist.all_gather(allv, v)
                allv = torch.stack(allv)
                if rank == 0:
                    rec = dict(T=T, routing=routing, world=world, mode=m['name'], beta=args.beta,
                               work=WORK, pieces=NP,
                               layer_us_median=round(graph_us[m['name']]['median'], 1),
                               layer_us_max_rank=round(graph_us[m['name']]['max'], 1),
                               peak_MB_max=round(allv[:, 0].max().item() / 1e6, 1),
                               rows_used_max=int(allv[:, 1].max()), rows_used_mean=round(allv[:, 1].mean().item()),
                               rows_alloc=int(allv[0, 2]), rows_borrowed_max=int(allv[:, 3].max()),
                               layers=NL, host_tail_MB=round(EV * 2 / 1e6, 1) if m['seg'] else 0.0,
                               weights_MB=round((NL * (W13E + W2E) - ((NL - 1) * EV if m['seg'] else 0)) * 2 / 1e6, 1),
                               total_MB=round(allv[:, 0].max().item() / 1e6 + (NL * (W13E + W2E) - ((NL - 1) * EV if m['seg'] else 0)) * 2 / 1e6, 1),
                               bitwise_equal=bool(eq.item()), all_rows_placed=bool(fits.item()), ref_rel_err=round(err.item(), 5),
                               phase='profile' if profiling else 'serve', pull=args.pull, cond=args.cond, kv_profile=args.kv_profile,
                               kv_pool_MB_min=round(allv[:, 4].min().item() / 1e6, 1),
                               kv_tokens_min=int(allv[:, 4].min().item()) // KV_TOKEN_BYTES,
                               reserve_MB_max=round(allv[:, 5].max().item() / 1e6, 1),
                               compared_to_ref=bool(args.compare_to and not profiling))
                    print(json.dumps(rec), flush=True)
                    if args.out:
                        with open(args.out, 'a') as f:
                            f.write(json.dumps(rec) + '\n')
        buf.destroy()
        torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--world', type=int, default=4)
    p.add_argument('--tokens', default='512,1024,2048,4096')
    p.add_argument('--routing', default='uniform')
    p.add_argument('--beta', type=float, default=1.25)
    p.add_argument('--graph-iters', type=int, default=6, help='even: layers alternate, a replay ends where it began')
    p.add_argument('--layers', type=int, default=2, help='layers with their own weights, run in turn')
    p.add_argument('--host-mb', type=float, default=0, help='MB of each layer\'s GEMM 1 weights kept in pinned host memory (segment mode, epa PagedWeights)')
    p.add_argument('--page-mb', type=float, default=16, help='MB per host-to-GPU page copy')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--port', type=int, default=29691)
    p.add_argument('--out', default='')
    p.add_argument('--profile', type=int, default=0)
    p.add_argument('--work', type=int, default=0, help='rows of the work buffer for borrowed rows (0: no borrowing)')
    p.add_argument('--pieces', type=int, default=0, help='borrowed pieces of WORK rows each (0 with --work: enough to fill the leased h)')
    p.add_argument('--cond', type=int, default=1,
                   help='--pull gemm*, graph runs: skip empty pieces with CUDA conditional IF nodes. 1: a nested IF per '
                        'piece; 2: one IF ("piece 0 has rows") around all pieces; 0: off')
    p.add_argument('--pf-from', default='g2', choices=['g1', 'g2'],
                   help='--pull gemm: piece q >= 2 is prefetched by GEMM 1 of piece q - 1 (g1) or GEMM 2 of piece q - 2 (g2)')
    p.add_argument('--pull', default='gemm', choices=['gemm', 'gemm-nopf', 'kernel'],
                   help='borrowed rows: GEMM 1 pulls them, earlier GEMMs prefetch (epa::RowPull, 2 work buffers); gemm-nopf: no prefetch (1 buffer); kernel: a copy kernel (2 buffers)')
    p.add_argument('--modes', default='upstream,cpusync,segment', help='which of upstream, cpusync, segment to run')
    p.add_argument('--budget-mb', type=float, default=0, help='per-GPU memory for the layer above the weights (0: no limit); a ballast takes the rest')
    p.add_argument('--slack-mb', type=float, default=256, help='extra room under a budget for CUDA runtime workspaces')
    p.add_argument('--kv-reserve-mb', type=float, default=0, help='serving fill: a KV-cache pool takes all free memory but this reserve for the layer (MB, segment buffers included)')
    p.add_argument('--kv-profile', default='', help='routing of a profile pass (like vLLM / SGLang); then a KV pool takes all but what that pass took + --kv-margin-mb, and --routing is served')
    p.add_argument('--kv-margin-mb', type=float, default=64, help='extra reserve on top of the profiled need')
    p.add_argument('--save-out', default='', help='save each rank\'s combined output to <prefix>.T<T>.<routing>.r<rank>.pt')
    p.add_argument('--compare-to', default='', help='check outputs are bitwise equal to a run saved with --save-out')
    a = p.parse_args()
    a.tokens = [int(t) for t in a.tokens.split(',')]
    assert a.graph_iters % a.layers == 0, 'graph-iters must be a multiple of --layers (a replay ends where it began)'
    a.routing = a.routing.split(',')
    a.modes = a.modes.split(',')
    mp.spawn(worker, args=(a.world, a), nprocs=a.world)
