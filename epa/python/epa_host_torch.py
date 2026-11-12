"""epa host tier (include/epa/host_tier.hpp) for PyTorch callers: PagedWeights.

N layers of read-only weights, each a contiguous tensor on the GPU, of which only one shared
tail is resident: page_in(k) copies layer k's tail from pinned host memory into it on the current
stream (CUDA-graph capturable: plain host-to-device copies from pinned memory). HBM saved:
(N - 1) x tail_bytes. The caller orders page_in(k) after every reader of the previous layer's tail
and makes layer k's readers wait for it.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch

_INCLUDE = Path(__file__).resolve().parents[1] / 'include'

_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <epa/host_tier.hpp>

struct PagedWeightsT {
    epa::PagedWeights w;
    PagedWeightsT(int64_t device, int64_t layers, int64_t layer_bytes, int64_t tail_bytes) {
        cuInit(0);
        w.create(static_cast<int>(device), static_cast<int>(layers), layer_bytes, tail_bytes);
    }
    ~PagedWeightsT() { w.destroy(); }
    // layer k's contents from a device tensor of layer_bytes (setup)
    void load(int64_t k, torch::Tensor src) {
        TORCH_CHECK(src.is_cuda() and src.is_contiguous() and src.numel() * src.element_size() == w.layer_bytes);
        w.load(static_cast<int>(k), src.data_ptr(), c10::cuda::getCurrentCUDAStream().stream());
    }
    // layer k as a tensor of `shape` / `dtype` (no copy; complete only once its tail is paged in)
    torch::Tensor view(int64_t k, std::vector<int64_t> shape, torch::Dtype dtype) {
        return torch::for_blob(w.layer(static_cast<int>(k)), shape)
            .options(torch::TensorOptions().dtype(dtype).device(torch::kCUDA, w.device))
            .target_device(torch::Device(torch::kCUDA, w.device)).make_tensor();
    }
    void page_in(int64_t k, int64_t page_bytes) {
        w.page_in(static_cast<int>(k), c10::cuda::getCurrentCUDAStream().stream(), page_bytes);
    }
    int64_t tail_bytes() const { return w.tail_bytes; }
    int64_t saved_bytes() const { return w.saved_bytes(); }
    int64_t hbm_bytes() const { return w.hbm_bytes(); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<PagedWeightsT>(m, "PagedWeights")
        .def(pybind11::init<int64_t, int64_t, int64_t, int64_t>())
        .def("load", &PagedWeightsT::load)
        .def("view", &PagedWeightsT::view)
        .def("page_in", &PagedWeightsT::page_in)
        .def("tail_bytes", &PagedWeightsT::tail_bytes)
        .def("saved_bytes", &PagedWeightsT::saved_bytes)
        .def("hbm_bytes", &PagedWeightsT::hbm_bytes);
}
'''


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load_inline
    return load_inline(name='epa_host_torch', cpp_sources='', cuda_sources=_SRC,
                       extra_include_paths=[str(_INCLUDE)], extra_cuda_cflags=['-O3', '-std=c++17'],
                       extra_ldflags=['-lcuda'], verbose=False)


def paged_weights(layers: list[torch.Tensor], tail_bytes: int):
    """PagedWeights holding `layers` (same shape, contiguous, on one GPU); returns (pw, views)."""
    shape, dtype, dev = list(layers[0].shape), layers[0].dtype, layers[0].device
    nbytes = layers[0].numel() * layers[0].element_size()
    pw = _ext().PagedWeights(dev.index, len(layers), nbytes, tail_bytes)
    for k, t in enumerate(layers):
        pw.load(k, t)
    torch.cuda.synchronize(dev)
    return pw, [pw.view(k, shape, dtype) for k in range(len(layers))]
