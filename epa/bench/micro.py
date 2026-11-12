"""E1: cost of EPArena's primitives on the 8-GPU NVLink node.

  push vs pull of message rows (the tier-1 vs tier-3 data path)
  NVLink fetch-add round trip and reservation throughput (tier 2)

usage: python micro.py [--out results/micro.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import epa_build  # noqa: E402

ROWS = 65536
HEAP = 2 * ROWS * 14 * 1024 + (1 << 20)


def worker(rank, world, args):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29611')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    ext = epa_build.load('epa_micro', [epa_build.ROOT / 'csrc/micro.cu'])
    epa_build.init_heap(ext, HEAP)
    res = {}

    # push vs pull, all ranks at once (all-to-all traffic), several row sizes
    for row_bytes in (7424, 14352, 28672):
        n = min(ROWS, (HEAP // 2 - (1 << 20)) // row_bytes)
        for blocks, threads in ((32, 256), (64, 512), (132, 512)):
            for pull in (False, True):
                dist.barrier()
                ms = ext.copy_rows(pull, 0, HEAP // 2, n, row_bytes, blocks, threads, 20)
                gbs = n * row_bytes / ms / 1e6
                t = torch.tensor([gbs], device='cuda')
                dist.all_reduce(t, op=dist.ReduceOp.MIN)
                res[f'{"pull" if pull else "push"}/row{row_bytes}/sm{blocks}x{threads}'] = t.item()

    # fetch-add latency: rank 0 to rank 1, others idle
    dist.barrier()
    if rank == 0:
        res['atomic_rtt_ns_peer'] = ext.atomic_latency(HEAP - 4096, 1, 2000)
        res['atomic_rtt_ns_local'] = ext.atomic_latency(HEAP - 4096, 0, 2000)
    dist.barrier()

    # reservation throughput: all ranks, every warp reserves 1 row per lane per iteration
    for blocks in (16, 64, 132):
        for agg in (False, True):
            dist.barrier()
            iters = 64
            ms = ext.reserve_rate(agg, HEAP - 8192, blocks, 256, iters)
            lane_reqs = blocks * 256 * iters
            res[f'reserve/{"warp" if agg else "lane"}/sm{blocks}'] = dict(us=ms * 1e3, mreq_per_s=lane_reqs / ms / 1e3)

    if rank == 0:
        print(json.dumps(res, indent=1))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    dist.barrier()
    ext.heap_free()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(Path(__file__).parent / 'results/micro.json'))
    p.add_argument('--world', type=int, default=8)
    a = p.parse_args()
    epa_build.load('epa_micro', [epa_build.ROOT / 'csrc/micro.cu'])   # build once before spawning
    mp.spawn(worker, args=(a.world, a), nprocs=a.world)
