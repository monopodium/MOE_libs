"""Debug: per-SM globaltimer trace of one EPA dispatch (built with -DEPA_TRACE)."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.argv += [] ; sys.path.insert(0, os.path.dirname(__file__))
import bench_mc as B
import epa_build

def build():
    return epa_build.load('epa_mooncake_trace', [B.HERE / 'harness.cu'],
                          includes=[B.HERE, B.MC / 'mooncake-ep/include', B.MC / 'mooncake-transfer-engine/include'],
                          defines=['USE_CUDA', 'EPA_TRACE'])

def worker(rank, world, cap, pool, routing):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29655')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', rank))
    ext = build(); T = 256
    gen = torch.Generator(device='cuda').manual_seed(rank)
    topk, w = B.make_routing(T, routing, gen)
    x = torch.randn((T, B.H), generator=gen, device='cuda').to(torch.bfloat16)
    inst = B.Inst(ext, True, T, cap, pool, False, rank, world)
    o = B.outputs(T, world, False)
    y = torch.zeros((B.E // world, world * T, B.H), dtype=torch.bfloat16, device='cuda')
    out = torch.empty((T, B.H), dtype=torch.bfloat16, device='cuda')
    for _ in range(10):
        B.run_pair(ext, inst, x, topk, w, o, y, out)
    torch.cuda.synchronize(); dist.barrier()
    ext.dispatch(inst.id, x, topk, o['recv_x'], o['scales'], o['src_info'], o['layout'], o['count'])
    torch.cuda.synchronize()
    t = ext.trace()[:48].double()
    t0 = t[:, 0].min()
    rel = (t - t0) / 1e3
    names = ['start', 'count_begin', 'scan_done', 'published', 'sends_done', 'recv_begin', 'end']
    if rank in (0, 3):
        print(f'rank {rank} cap={cap} pool={pool} {routing}: ' + ', '.join(
            f'{n} med {rel[:, i].median():.1f} max {rel[:, i].max():.1f}' for i, n in enumerate(names)), flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    cap, pool, routing = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    build()
    mp.spawn(worker, args=(8, cap, pool, routing), nprocs=8)
