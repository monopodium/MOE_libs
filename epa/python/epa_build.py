"""Build helpers shared by EPArena's benchmarks and library harnesses."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_HOME = os.environ.setdefault('CUDA_HOME', '/home/Shuang/envs/cuda132')
os.environ['PATH'] = f'{CUDA_HOME}/bin:' + os.environ.get('PATH', '')
os.environ.setdefault('CC', '/usr/bin/gcc')
os.environ.setdefault('CXX', '/usr/bin/g++')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '10.3a')
BUILD = Path(os.environ.get('EPA_BUILD_DIR', '/run/user/1008/epa_build'))


def load(name, sources, includes=(), defines=(), verbose=False):
    from torch.utils.cpp_extension import load as _load
    d = BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    flags = ['-O3', '-std=c++20', '--expt-relaxed-constexpr', '-lineinfo',
             '-gencode=arch=compute_103a,code=sm_103a'] + [f'-D{x}' for x in defines]
    return _load(name=name, sources=[str(s) for s in sources],
                 extra_include_paths=[str(ROOT / 'include')] + [str(i) for i in includes],
                 extra_cuda_cflags=flags, extra_cflags=['-O3', '-std=c++20'] + [f'-D{x}' for x in defines],
                 extra_ldflags=['-lcuda'], build_directory=str(d), verbose=verbose)


def init_heap(ext, nbytes, group=None):
    """Create this rank's heap and map every peer's heap (cudaIpc)."""
    import torch
    import torch.distributed as dist
    ext.heap_create(nbytes)
    mine = ext.heap_handle()
    allh = [torch.empty_like(mine) for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(allh, mine, group=group)
    ext.heap_open(torch.stack(allh), dist.get_rank(group))
    torch.cuda.synchronize()
    dist.barrier(group)
