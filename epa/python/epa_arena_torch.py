"""epa::Arena page operations as CUDA-graph-capturable PyTorch calls (one tiny kernel each),
for Python-orchestrated libraries such as dist_moe. Built once with
torch.utils.cpp_extension from epa/include/epa/arena.cuh.

The arena lives inside a caller-owned uint8 buffer: `num_pages` pages of `page_bytes`
at byte `data_start` of the buffer and the ring state at byte `meta_start`.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch

_INCLUDE = Path(__file__).resolve().parents[1] / 'include'

_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <epa/arena.cuh>

static epa::Arena arena_of(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes) {
    epa::Arena a;
    a.base = static_cast<char*>(buf.data_ptr()) + meta_start;
    a.page_bytes = page_bytes;
    a.num_pages = static_cast<int>(num_pages);
    a.data_off = data_start - meta_start;
    return a;
}

// Take one page if rows[0] > 0 (waiting for one to come back if none is free); write
// the byte offsets in `buf` of the page plus each relative offset rel[i] - rel[0].
__global__ void round_take_kernel(epa::Arena a, int64_t data_start, const int* rows,
                                  const int64_t* off0, const int64_t* off1, const int64_t* off2,
                                  int64_t* out, int* page) {
    int id = -1;
    if (rows[0] > 0) id = a.page_at(a.wait_take(1), 0);
    *page = id;
    const int64_t at = data_start + static_cast<int64_t>(id < 0 ? 0 : id) * a.page_bytes;
    out[0] = at;
    out[1] = at + (off1[0] - off0[0]);
    out[2] = at + (off2[0] - off0[0]);
}
__global__ void round_give_kernel(epa::Arena a, const int* page) {
    if (*page >= 0) a.give(*page);
}

void init(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes) {
    auto a = arena_of(buf, meta_start, data_start, num_pages, page_bytes);
    epa::arena_init_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
}
void round_take(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes,
                torch::Tensor rows, torch::Tensor off0, torch::Tensor off1, torch::Tensor off2,
                torch::Tensor out, torch::Tensor page) {
    auto a = arena_of(buf, meta_start, data_start, num_pages, page_bytes);
    round_take_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        a, data_start, rows.data_ptr<int>(), off0.data_ptr<int64_t>(), off1.data_ptr<int64_t>(),
        off2.data_ptr<int64_t>(), out.data_ptr<int64_t>(), page.data_ptr<int>());
}
void round_give(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes,
                torch::Tensor page) {
    auto a = arena_of(buf, meta_start, data_start, num_pages, page_bytes);
    round_give_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(a, page.data_ptr<int>());
}
std::vector<int64_t> stats(torch::Tensor buf, int64_t meta_start) {
    epa::ArenaMeta m;
    cudaMemcpy(&m, static_cast<char*>(buf.data_ptr()) + meta_start, sizeof m, cudaMemcpyDeviceToHost);
    return {m.free_pages, m.min_free_pages, static_cast<int64_t>(m.taken)};
}
'''


_DECL = r'''
#include <torch/extension.h>
void init(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes);
void round_take(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes,
                torch::Tensor rows, torch::Tensor off0, torch::Tensor off1, torch::Tensor off2,
                torch::Tensor out, torch::Tensor page);
void round_give(torch::Tensor buf, int64_t meta_start, int64_t data_start, int64_t num_pages, int64_t page_bytes,
                torch::Tensor page);
std::vector<int64_t> stats(torch::Tensor buf, int64_t meta_start);
'''


@functools.cache
def ext():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name='epa_arena_torch', cpp_sources=_DECL, cuda_sources=_SRC,
        functions=['init', 'round_take', 'round_give', 'stats'],
        extra_include_paths=[str(_INCLUDE)], extra_cuda_cflags=['-O3', '-std=c++20'], verbose=False)


def meta_bytes(num_pages: int) -> int:
    """Bytes of the ring state (epa::Arena::meta_bytes)."""
    return -(-(256 + num_pages * 12) // 4096) * 4096
