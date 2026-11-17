#pragma once

#include <c10/util/accumulate.h>
#include <cuda_runtime.h>
#include <memory>
#include <numeric>
#include <vector>
#include <pybind11/functional.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/layout/ep/epa.cuh>
#include <deep_ep/layout/ep/epa_out.cuh>
#include <deep_ep/layout/ep/epa_seg.cuh>
#include <deep_ep/layout/ep/token.cuh>
#include <deep_ep/layout/ep/workspace.cuh>
#include <deep_ep/layout/ep/eplb.cuh>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../kernels/ep/api.hpp"
#include "../runtime/jit.hpp"
#include "../utils/event.hpp"
#include "../utils/tensor.hpp"

namespace deep_ep::ep {

class EPBuffer: public BufferBase {
    // Buffer bytes exclude workspace
    // Memory layout: [Workspace, buffer]
    int64_t num_buffer_bytes;

    // EPArena: device table of every NVLink peer's EP buffer base (for tier-3 pulls)
    mutable void** epa_peer_buffers = nullptr;

    // Paged expanded output (layout/ep/epa_out.cuh), allocated at the first paged dispatch:
    // the arena whose data region is recv_x, its meta block, and y / weights with the same rows
    mutable int epa_out_p = 0, epa_out_pages = 0;
    mutable int epa_out_keys = 0;          // pages are keyed by local expert (expanded) or by source rank (normal)
    mutable bool epa_out_normal = false;   // paged normal layout: combine reads y through recv_src_metadata[:, 2]
    mutable int* epa_out_meta = nullptr;
    mutable void* epa_out_arena = nullptr;
    mutable torch::Tensor epa_out_x, epa_out_y, epa_out_w;

    // EPArena segment: header at buffer + bytes - region, rows right after it. (Re)initialised
    // when C changes; call epa_seg_init on every rank and barrier before the first dispatch.
    int* epa_seg_setup(const int& rows, const int64_t& row_bytes) const {
        const int64_t region = math::align<int64_t>(layout::EPASeg::kHeaderBytes + int64_t(rows) * row_bytes, 2 << 20);
        EP_HOST_ASSERT(region < num_buffer_bytes and "EP_EPA_SEG_ROWS: the EP buffer has no room for the segment (set EP_EPA_SEG_BYTES)");
        int* hd = reinterpret_cast<int*>(math::advance_ptr(context->buffer, num_buffer_bytes - region));
        if (epa_seg_rows != rows or epa_seg_row_bytes != row_bytes) {
            std::vector<int> h(layout::EPASeg::kHeaderBytes / 4, 0);
            h[0] = rows;
            auto* bases = reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(h.data()) + 512);
            for (int r = 0; r < context->num_nvl_ranks; ++ r)
                bases[r] = reinterpret_cast<unsigned long long>(context->get_sym_ptr(
                    math::advance_ptr(context->buffer, num_buffer_bytes - region + layout::EPASeg::kHeaderBytes), r));
            CUDA_RUNTIME_CHECK(cudaMemcpy(hd, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            epa_seg_hd = hd, epa_seg_rows = rows, epa_seg_row_bytes = row_bytes;
        }
        return hd;
    }
    // Peer spill (EP_EPA_SPILL=1, shared pool): every NVLink peer's allocation (workspace + EP buffer)
    // mapped again into one VA range at a fixed stride, so recv_x / y are one view over all GPUs' pages
    mutable void* epa_window = nullptr;
    // EPArena segment (EP_EPA_SEG_ROWS): header + rows at the tail of the EP buffer
    mutable int* epa_seg_hd = nullptr;
    mutable int epa_seg_rows = 0;
    mutable int64_t epa_seg_row_bytes = 0;
    mutable int64_t epa_window_stride = 0, epa_window_map_bytes = 0;

    // Host workspace
    void *host_workspace, *mapped_host_workspace;

    // Whether to use hybrid mode (scale-out with scale-up)
    bool allow_hybrid_mode;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    bool prefer_overlap_with_compute;

    // Some EP hybrid mode settings
    static constexpr int kNumMaxChannelsPerSM = 8;
    static constexpr int kNumMaxSMs = 160;
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;

public:
    std::shared_ptr<comm::Context> context;

    // For load balance
    torch::Tensor lb_storage;

    EPBuffer(const int& rank_idx, const int& num_ranks,
             const int64_t& nccl_comm,
             const int64_t& num_buffer_bytes,
             const int64_t& num_lb_buffer_bytes,
             const bool& allow_hybrid_mode,
             const bool& allow_multiple_reduction,
             const bool& prefer_overlap_with_compute,
             const std::optional<int>& sl_idx, const int& num_allocated_qps,
             const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
             const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        num_buffer_bytes(num_buffer_bytes),
        allow_hybrid_mode(allow_hybrid_mode),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute) {
        EP_HOST_ASSERT(num_buffer_bytes > 0 and num_buffer_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_lb_buffer_bytes >= 0 and num_lb_buffer_bytes % kNumAllocationAlignmentBytes == 0);

        // Workspace is aligned to 2 MB so that it sits cleanly at the front of the GPU segment
        const auto num_workspace_bytes = math::align<int64_t>(
            layout::EPWorkspaceLayout::get_num_bytes(), kNumAllocationAlignmentBytes);

        context = std::make_shared<comm::Context>(
            nccl_comm, symmetric::shared_comm_t{}, num_ranks, rank_idx,
            num_workspace_bytes, num_buffer_bytes + num_lb_buffer_bytes, 0, true,
            allow_hybrid_mode, sl_idx, num_allocated_qps,
            0, num_cpu_timeout_secs, num_gpu_timeout_secs);
        main_context = context;
        auto& workspace = *static_cast<layout::EPSignals*>(context->workspace);
        context->set_barrier_signals(&workspace.barrier_signals);

        // Expose only the LB region; the preceding bytes remain reserved for EP communication.
        lb_storage = torch::from_blob(
            context->buffer, {num_buffer_bytes + num_lb_buffer_bytes}, [context = context](void*) {},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        ).narrow(0, num_buffer_bytes, num_lb_buffer_bytes);

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::EPWorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        std::memset(host_workspace, 0, layout::EPWorkspaceLayout::get_num_bytes());

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
    }

    ~EPBuffer() noexcept(false) override {
        destroy_on_destruction("EP");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);

        // Finish all works on all GPUs
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

        // Destroy NCCL context
        context->finalize();

        // Destroy load-balance storage
        lb_storage = torch::Tensor();

        // Paged expanded output
        epa_out_x = epa_out_y = epa_out_w = torch::Tensor();
        if (epa_out_meta != nullptr) {
            CUDA_RUNTIME_CHECK(cudaFree(epa_out_meta));
            CUDA_RUNTIME_CHECK(cudaFree(epa_out_arena));
            epa_out_meta = nullptr, epa_out_arena = nullptr;
        }

        // Cannot use anymore
        destroyed = true;
    }

    static torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event) {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto comm_stream = comm::get_comm_stream();

        // Wait previous tasks to finish
        if (previous_event.has_value()) {
            comm::stream_wait(comm_stream, previous_event.value());
        } else {
            comm::stream_wait(comm_stream, compute_stream);
        }
        return compute_stream;
    }

    static torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                           const bool& allocate_on_comm_stream) {
        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        if (previous_event.has_value())
            EP_HOST_ASSERT(allocate_on_comm_stream);

        // Allocate all tensors on communication stream if set
        // NOTES: do not allocate tensors upfront!
        const auto compute_stream = stream_control_prologue(previous_event);
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(comm::get_comm_stream());
        return compute_stream;
    }

    static std::optional<EventHandle> stream_control_epilogue(const tensor_list_t& tensors,
                                                              const at::cuda::CUDAStream& compute_stream,
                                                              const bool& allocate_on_comm_stream,
                                                              const bool& async_with_compute_stream) {
        // Ensure memory access safety between two streams
        const auto comm_stream = comm::get_comm_stream();
        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            event = EventHandle(comm_stream);

            // NOTES: this environment only applies to V2 APIs
            if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
                event->tensors_to_record = tensors;
            } else {
                for (auto& t: tensors) if (t.has_value()) {
                    t->record_stream(compute_stream);
                    t->record_stream(comm_stream);
                }
            }
        } else {
            comm::stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // The CUDA event marking the finishing
        return event;
    }

    // EPArena on the direct NVLink dispatch path (see `layout/ep/epa.cuh`):
    //   EP_EPA_CAP   tier-1 slots per source rank (unset: upstream layout)
    //   EP_EPA_POOL  tier-2 pool rows shared by all source ranks
    // or the full epa::Arena (returned as cap = -(P + 1), pool = pages):
    //   EP_EPA_ARENA_P      tokens per page
    //   EP_EPA_ARENA_PAGES  pages in each rank's arena
    static std::pair<int, int> get_epa_config(const int& num_scaleout_ranks, const bool& is_scaleup_nvlink) {
        if (const int P = get_env<int>("EP_EPA_ARENA_P", 0); P > 0 and num_scaleout_ranks == 1 and is_scaleup_nvlink) {
            // Shared pool (EP_EPA_SHARED_A pages in zone A, EP_EPA_SHARED_B in zone B; EP_EPA_ARENA_P =
            // EP_EPA_OUT_P - 1): receive, recv_x and y pages in one page space, see layout/ep/epa.cuh
            if (const int na = get_env<int>("EP_EPA_SHARED_A", 0); na > 0)
                return {-(P + 1), -((na << 16) | get_env<int>("EP_EPA_SHARED_B", 0))};
            return {-(P + 1), get_env<int>("EP_EPA_ARENA_PAGES", 0)};
        }
        const int cap = get_env<int>("EP_EPA_CAP", -1);
        if (cap < 0 or num_scaleout_ranks != 1 or not is_scaleup_nvlink)
            return {-1, 0};
        return {cap, get_env<int>("EP_EPA_POOL", 0)};
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs, const int& num_topk,
                                            const int& elem_size,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                            const bool& is_scaleup_nvlink) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

        if (const auto [epa_cap, epa_pool] = get_epa_config(num_scaleout_ranks, is_scaleup_nvlink); epa_cap != -1) {
            // Direct dispatch with EPArena: window (or arena pages), home rows, descriptors
            return layout::EPALayout(token_layout.get_num_bytes<false, int64_t>(), num_ranks, num_max_tokens_per_rank,
                                     epa_cap, epa_pool, 0, static_cast<int64_t>(hidden) * elem_size).num_data_bytes();
        }
        if (num_scaleout_ranks == 1) {
            // Direct dispatch
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_ranks, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid dispatch
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, 1, num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleout_ranks,
                /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& is_scaleup_nvlink,
                                           const bool& allow_multiple_reduction) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct combine
            const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : num_ranks,
                // For single reduction cases, the maximum number of received tokens is
                // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
                num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid combine
            const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
            const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, allow_multiple_reduction ? 1 : num_topk,
                /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch,
                                         const bool& allow_hybrid_mode,
                                         const bool& allow_multiple_reduction) {
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Topology
        const auto [num_rdma_ranks, num_nvl_ranks] = comm::get_physical_domain_size(nccl_comm);
        const auto [num_scaleout_ranks, num_scaleup_ranks] = comm::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
        const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

        // Dispatch size
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0; // An approximation for number of SF packs
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink);

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink, allow_multiple_reduction);

        // Return the maximum of those layouts (plus EPArena's tables at the tail), aligned to 2 MB
        const auto [epa_cap_, epa_pool_] = get_epa_config(num_scaleout_ranks, is_scaleup_nvlink);
        const auto num_epa_table_bytes = epa_cap_ != -1 ?
            layout::EPALayout::num_table_bytes(num_scaleout_ranks * num_scaleup_ranks, epa_cap_, epa_pool_, num_max_tokens_per_rank) : 0;
        return math::align<int64_t>(std::max(num_dispatch_bytes, num_combine_bytes) + num_epa_table_bytes,
                                    kNumAllocationAlignmentBytes);
    }

    pybind11::tuple
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf,
             const bool& defer_epilogue) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        EP_HOST_ASSERT((num_sms > 1 or context->num_scaleout_ranks == 1 or context->num_scaleup_ranks == 1) and
                       "Hybrid dispatch requires at least 2 SMs");

        // Zero padding only makes sense with expand mode
        EP_HOST_ASSERT(not do_zero_padding or do_expand);

        // Cached mode must have responding handles
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(cached_num_recv_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_num_expanded_tokens.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());
            EP_HOST_ASSERT(cached_recv_src_metadata.has_value());
            EP_HOST_ASSERT(cached_recv_src_metadata->is_cuda() and cached_recv_src_metadata->is_contiguous());

            // Hybrid kernels require more
            if (context->num_scaleout_ranks > 1) {
                EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());
                EP_HOST_ASSERT(cached_channel_linked_list.has_value());
            }
        }

        // Check data tensor
        const auto [num_tokens, hidden] = get_shape<2>(x);
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        const auto num_local_experts = num_experts / context->num_ranks;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // Check SF stuffs
        int num_sf_packs = 0;
        void* sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF must be FP32 or packed UE8M0x4
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            EP_HOST_ASSERT(sf->is_cuda());
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            num_sf_packs = num_sf_packs_;
            sf_ptr = sf->data_ptr();
            sf_token_stride = sf->stride(0);
            sf_hidden_stride = sf->stride(1);
        }

        // Check top-k stuffs
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream);
        const auto comm_stream = comm::get_comm_stream();

        // The number of received tokens per expert
        // This is useful for expanding mode
        EP_HOST_ASSERT(num_experts % context->num_ranks == 0);
        auto psum_num_recv_tokens_per_expert = cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(psum_num_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts + 1}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // The unaligned (actual) number of received tokens per expert
        // Written by the dispatch kernel's notify warps, used by the epilogue for zero padding
        auto num_unaligned_recv_tokens_per_expert = cached_num_unaligned_recv_tokens_per_expert.value_or(torch::Tensor());
        int* num_unaligned_recv_tokens_per_expert_ptr = nullptr;
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(num_unaligned_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.is_cuda() and num_unaligned_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            num_unaligned_recv_tokens_per_expert = torch::empty(
                {num_local_experts}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }
        num_unaligned_recv_tokens_per_expert_ptr = num_unaligned_recv_tokens_per_expert.data_ptr<int>();

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        auto psum_num_recv_tokens_per_scaleup_rank = cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());
        if (cached_mode) {
            const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
            EP_HOST_ASSERT(num_scaleup_ranks == context->num_scaleup_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        } else {
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {context->num_scaleup_ranks}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // Decide number of channels by shared memory consumption
        // Only for hybrid version
        int num_channels_per_sm = 1, num_channels = 1;
        const int num_smem_bytes = jit->device.get_num_smem_bytes();
        if (context->num_scaleout_ranks > 1) {
            const auto dispatch_token_layout = get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk);
            const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
            EP_HOST_ASSERT(num_sms <= kNumMaxSMs);
            num_channels_per_sm = std::min<int>(
                (num_smem_bytes - get_num_notify_smem_bytes(context->num_ranks, num_experts)) / dispatch_token_layout.get_num_bytes<true>(),
                32 - kNumNotifyWarps);
            num_channels_per_sm = std::min<int>(
                num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
                num_channels_per_sm);
            num_channels_per_sm = std::min<int>(
                /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
            if (not prefer_overlap_with_compute)
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
            num_channels = num_sms * num_channels_per_sm;
            if (get_env<int>("EP_BUFFER_DEBUG"))
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        }

        // Non-hybrid mode handles
        auto dst_buffer_slot_idx = cached_dst_buffer_slot_idx.value_or(torch::Tensor());
        if (context->num_scaleout_ranks == 1) {
            if (cached_mode) {
                const auto [num_tokens__, num_topk_] = get_shape<2>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                // Allocate a new tensor
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            }
        }

        // Hybrid mode handles
        std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;
        int *token_metadata_at_forward_ptr = nullptr, *channel_linked_list_ptr = nullptr;
        if (context->num_scaleout_ranks > 1) {
            // The token destination slot idx during forward
            // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th token's index in the l-th rank buffer
            // NOTES: Used primarily for cached mode
            // TODO: May make it a linked list to remove the redundant info in `token_metadata_at_forward`
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
            if (cached_mode) {
                const auto [num_channels_, num_scaleout_ranks_, num_max_tokens_per_channel_, num_topk_] =
                    get_shape<4>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_channels == num_channels_ and context->num_scaleout_ranks == num_scaleout_ranks_ and
                               num_max_tokens_per_channel == num_max_tokens_per_channel_ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                dst_buffer_slot_idx = torch::empty(
                    {num_channels, context->num_scaleout_ranks, num_max_tokens_per_channel, num_topk},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }

            // The token metadata during forward
            // `[i, j]` means: in channel i, the j-th forwarded token's metadata
            // Info contains:
            //   - Scaleout rank index and source token index in the original rank (0)
            //   - Whether the token is the last one in the chunk (1)
            //   - cached top-k scaleup peer indices (top-k)
            //   - each selections' destination slot indices (top-k)
            const auto num_max_forwarded_tokens = context->num_scaleout_ranks * num_max_tokens_per_channel + 1;
            const auto num_forward_metadata_dims = 2 + num_topk * 2;
            if (cached_mode) {
                token_metadata_at_forward = cached_token_metadata_at_forward;
                const auto [num_channels_, num_max_forwarded_tokens_, num_forward_metadata_dims_] = get_shape<3>(token_metadata_at_forward.value());
                EP_HOST_ASSERT(num_channels == num_channels_ and num_max_forwarded_tokens == num_max_forwarded_tokens_
                               and num_forward_metadata_dims == num_forward_metadata_dims_);
                EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
                EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);
            } else {
                token_metadata_at_forward = torch::empty(
                    {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();

            // Per-scaleup-peer-per-channel linked list
            // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
            if (cached_mode) {
                channel_linked_list = cached_channel_linked_list;
                const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
                channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
                EP_HOST_ASSERT(num_channels == num_channels__);
                EP_HOST_ASSERT(d1_ == context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
                EP_HOST_ASSERT(d2_ == context->num_scaleup_ranks);
                EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
                EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
            } else {
                channel_linked_list = torch::empty(
                    // Index 0 of the list means the starting item
                    {num_channels,
                    context->num_scaleout_ranks * num_max_tokens_per_channel + 1,
                    context->num_scaleup_ranks},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
        }

        // EPArena: tables live at the tail of the buffer, zeroed once; peers' buffer bases for pulls
        const auto [epa_cap, epa_pool] = get_epa_config(context->num_scaleout_ranks, context->is_scaleup_nvlink);
        const int64_t epa_tables = epa_cap != -1 ?
            num_buffer_bytes - layout::EPALayout::num_table_bytes(context->num_ranks, epa_cap, epa_pool, num_max_tokens_per_rank) : 0;
        if (epa_cap != -1 and epa_peer_buffers == nullptr) {
            std::vector<void*> peers(context->num_nvl_ranks);
            for (int r = 0; r < context->num_nvl_ranks; ++ r)
                peers[r] = context->get_sym_ptr(context->buffer, r);
            CUDA_RUNTIME_CHECK(cudaMalloc(&epa_peer_buffers, sizeof(void*) * peers.size()));
            CUDA_RUNTIME_CHECK(cudaMemcpy(epa_peer_buffers, peers.data(), sizeof(void*) * peers.size(), cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(math::advance_ptr(context->buffer, epa_tables), 0,
                layout::EPALayout::num_table_bytes(context->num_ranks, epa_cap, epa_pool, num_max_tokens_per_rank), comm_stream));
            if (epa_cap < -1) {
                // epa::Arena: every page free; page tables empty
                const auto epa = layout::EPALayout(
                    get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk).get_num_bytes<false, int64_t>(),
                    context->num_ranks, num_max_tokens_per_rank, epa_cap, epa_pool, epa_tables,
                    static_cast<int64_t>(hidden) * x.element_size());
                const auto img = epa::Arena::host_init_image(epa.num_pages);
                CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
                CUDA_RUNTIME_CHECK(cudaMemcpy(math::advance_ptr(context->buffer, epa.arena_offset()), img.data(), img.size(),
                                              cudaMemcpyHostToDevice));
                if (epa.shared) {
                    // Shared pool: zone B's ring, every page free; zone A must cover combine's layout
                    const auto img_b = epa::Arena::host_init_image(epa.num_pages_b);
                    CUDA_RUNTIME_CHECK(cudaMemcpy(math::advance_ptr(context->buffer, epa.ringb_offset()), img_b.data(), img_b.size(),
                                                  cudaMemcpyHostToDevice));
                    const auto combine_bytes = get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                        context->num_scaleout_ranks, context->num_scaleup_ranks, context->is_scaleup_nvlink, allow_multiple_reduction);
                    EP_HOST_ASSERT(static_cast<int64_t>(epa.num_pages) * epa.page_bytes() >= combine_bytes and
                                   "shared pool: zone A must cover combine's layout");
                }
            }
        }

        // Paged output: EP_EPA_OUT_P rows per page, EP_EPA_OUT_PAGES pages. Expanded layout: pages per
        // local expert; normal layout: pages per source rank (kernels get -P)
        const int out_p = get_env<int>("EP_EPA_OUT_P", 0);
        const bool shared_pool = layout::EPALayout::is_shared(epa_cap, epa_pool);
        const int out_pages = out_p <= 0 ? 0 : shared_pool ?
            layout::EPALayout::zone_a(epa_pool) + layout::EPALayout::zone_b(epa_pool) : get_env<int>("EP_EPA_OUT_PAGES", 0);
        if (out_p > 0) {
            EP_HOST_ASSERT(context->num_scaleout_ranks == 1 and "paged output needs the direct path");
            EP_HOST_ASSERT(not cached_mode and not do_zero_padding and not sf.has_value() and x.element_size() == 2);
            EP_HOST_ASSERT(out_pages > 0);
            if (epa_out_meta == nullptr) {
                const int L = do_expand ? num_local_experts : context->num_ranks;
                epa_out_keys = L, epa_out_normal = not do_expand;
                const int64_t page_bytes = static_cast<int64_t>(out_p) * hidden * x.element_size();
                if (shared_pool) {
                    // Shared pool: recv_x and y pages come from the EP buffer's page space (zones A and B)
                    const auto epa = layout::EPALayout(
                        get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk).get_num_bytes<false, int64_t>(),
                        context->num_ranks, num_max_tokens_per_rank, epa_cap, epa_pool, epa_tables, static_cast<int64_t>(hidden) * x.element_size());
                    if (epa.page_bytes() != page_bytes)
                        printf("shared pool page mismatch: layout %lld (rows %d, token %lld, row %lld, shared %d) vs out %lld\n",
                               (long long) epa.page_bytes(), epa.rows_per_page, (long long) epa.token_bytes, (long long) epa.row_bytes,
                               (int) epa.shared, (long long) page_bytes);
                    EP_HOST_ASSERT(epa.page_bytes() == page_bytes and "shared pool: EP_EPA_ARENA_P must be EP_EPA_OUT_P - 1");
                } else {
                    CUDA_RUNTIME_CHECK(cudaMalloc(&epa_out_arena, epa::Arena::bytes_for(out_pages, page_bytes)));
                    const auto img = epa::Arena::host_init_image(out_pages);
                    CUDA_RUNTIME_CHECK(cudaMemcpy(epa_out_arena, img.data(), img.size(), cudaMemcpyHostToDevice));
                }
                const auto n_ints = layout::EPAOutLayout::num_meta_ints(L, out_pages);
                CUDA_RUNTIME_CHECK(cudaMalloc(&epa_out_meta, n_ints * sizeof(int)));
                CUDA_RUNTIME_CHECK(cudaMemset(epa_out_meta, 0, n_ints * sizeof(int)));
                const auto out = layout::EPAOutLayout{epa_out_meta, L, out_pages, out_p};
                CUDA_RUNTIME_CHECK(cudaMemset(out.tile_expert(), 0xff, out_pages * sizeof(int)));
                int64_t rows = static_cast<int64_t>(out_pages) * out_p;
                if (shared_pool) {
                    // One view of the page space for both: a tile's recv_x page and y page differ
                    void* view = context->buffer;
                    if (get_env<int>("EP_EPA_SPILL", 0) > 0) {
                        // Peer spill: the view spans every GPU's buffer (page ids global: gpu * pps + local)
                        const int R = context->num_nvl_ranks;
                        const int64_t buf_off = static_cast<char*>(context->buffer) - static_cast<char*>(context->workspace);
                        if (epa_window == nullptr) {
                            epa_window_map_bytes = math::align<int64_t>(buf_off + num_buffer_bytes, kNumAllocationAlignmentBytes);
                            const int64_t unit = std::lcm<int64_t>(kNumAllocationAlignmentBytes, page_bytes);
                            epa_window_stride = math::align<int64_t>(epa_window_map_bytes, unit);
                            int dev = 0;
                            CUDA_RUNTIME_CHECK(cudaGetDevice(&dev));
                            CUdeviceptr w;
                            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&w, R * epa_window_stride, kNumAllocationAlignmentBytes, 0, 0));
                            for (int r = 0; r < R; ++ r) {
                                CUmemGenericAllocationHandle h;
                                CUDA_DRIVER_CHECK(lazy_cuMemRetainAllocationHandle(&h, context->get_sym_ptr(context->workspace, r)));
                                CUDA_DRIVER_CHECK(lazy_cuMemMap(w + r * epa_window_stride, epa_window_map_bytes, 0, h, 0));
                                CUDA_DRIVER_CHECK(lazy_cuMemRelease(h));
                                CUmemAccessDesc desc = {};
                                desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE, desc.location.id = dev;
                                desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                                CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(w + r * epa_window_stride, epa_window_map_bytes, &desc, 1));
                            }
                            epa_window = reinterpret_cast<void*>(w);
                            // The window must alias every GPU's buffer: compare its tables (tail) bytes
                            std::vector<uint8_t> a(4096), b(4096);
                            for (int r = 0; r < R; ++ r) {
                                const int64_t off = num_buffer_bytes - 4096;
                                bool same = false;
                                for (int tries = 0; tries < 5 and not same; ++ tries) {   // a peer may be writing its tables
                                    CUDA_RUNTIME_CHECK(cudaMemcpy(a.data(), math::advance_ptr(context->get_sym_ptr(context->buffer, r), off), 4096, cudaMemcpyDeviceToHost));
                                    CUDA_RUNTIME_CHECK(cudaMemcpy(b.data(), math::advance_ptr(epa_window, r * epa_window_stride + buf_off + off), 4096, cudaMemcpyDeviceToHost));
                                    same = a == b;
                                }
                                EP_HOST_ASSERT(same and "peer spill: window does not alias the peer buffer");
                            }
                        }
                        const int pps = static_cast<int>(epa_window_stride / page_bytes);
                        const int me = context->nvl_rank_idx;
                        const int hdr[2] = {me * pps, pps};
                        CUDA_RUNTIME_CHECK(cudaMemcpy(epa_out_meta + 12, hdr, sizeof(hdr), cudaMemcpyHostToDevice));
                        view = math::advance_ptr(epa_window, buf_off);
                        rows = static_cast<int64_t>(R) * pps * out_p;
                    }
                    // target_device: the window's first bytes are GPU 0's, so the device is not inferred
                    epa_out_x = torch::for_blob(view, {rows, hidden}).options(x.options()).target_device(x.device()).make_tensor();
                    epa_out_y = epa_out_x;
                } else {
                    epa_out_x = torch::from_blob(math::advance_ptr(epa_out_arena, epa::Arena::meta_bytes(out_pages)),
                                                 {rows, hidden}, x.options());
                    epa_out_y = torch::empty({rows, hidden}, x.options());
                }
                epa_out_w = torch::empty({rows}, torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat));
                epa_out_p = out_p, epa_out_pages = out_pages;
            }
            EP_HOST_ASSERT(epa_out_p == out_p and epa_out_pages == out_pages and epa_out_normal == not do_expand);
        }

        // EPArena segment (EP_EPA_SEG_ROWS rows, expanded layout, no CPU sync)
        int* seg_hd = nullptr;
        if (const int seg_rows = get_env<int>("EP_EPA_SEG_ROWS", 0);
            seg_rows > 0 and do_expand and not cached_mode and not do_cpu_sync and out_p <= 0) {
            seg_hd = epa_seg_setup(seg_rows, static_cast<int64_t>(hidden) * x.element_size());
            const int64_t seg_off = reinterpret_cast<char*>(seg_hd) - static_cast<char*>(context->buffer);
            EP_HOST_ASSERT(seg_off >= get_dispatch_buffer_size(num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),
                                                               context->num_scaleout_ranks, context->num_scaleup_ranks, context->is_scaleup_nvlink) and
                           seg_off >= get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk, context->num_scaleout_ranks,
                                                              context->num_scaleup_ranks, context->is_scaleup_nvlink, allow_multiple_reduction) and
                           "EP_EPA_SEG_ROWS: the segment overlaps dispatch / combine data");
        }

        // Check buffer size
        EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),
                       context->num_scaleout_ranks, context->num_scaleup_ranks,
                       context->is_scaleup_nvlink) <= (epa_cap != -1 ? epa_tables : num_buffer_bytes));

        // Ready and clean host workspace for this round
        const auto host_workspace_layout = layout::EPWorkspaceLayout(
            host_workspace,
            context->num_scaleout_ranks,
            context->num_scaleup_ranks,
            num_experts);
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), context->num_scaleup_ranks, 0);
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Do dispatch into the buffers (with SM limitation)
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
                        cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        num_unaligned_recv_tokens_per_expert_ptr,
                        dst_buffer_slot_idx.data_ptr<int>(),
                        token_metadata_at_forward_ptr,
                        num_tokens, num_max_tokens_per_rank,
                        hidden, x.element_size(),
                        num_sf_packs, sf_token_stride, sf_hidden_stride,
                        num_experts, num_topk, expert_alignment,
                        context->dev_comm, context->window,
                        context->buffer,
                        context->workspace, mapped_host_workspace,
                        context->scaleout_rank_idx, context->scaleup_rank_idx,
                        context->num_scaleout_ranks, context->num_scaleup_ranks,
                        context->is_scaleup_nvlink,
                        num_sms, num_channels_per_sm,
                        num_smem_bytes,
                        num_qps, context->num_gpu_timeout_cycles,
                        cached_mode, do_cpu_sync,
                        comm_stream,
                        epa_cap, epa_pool, epa_tables,
                        do_expand ? out_p : -out_p, out_pages, seg_hd != nullptr ? seg_hd : epa_out_meta, epa_out_arena);

        // For tensor recording
        tensor_list_t tensors_to_record = {
            x, sf, topk_idx, topk_weights,
            cumulative_local_expert_recv_stats,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list};

        // Epilogue can be deferred, so it is a lambda
        auto epilogue = [=, this](const at::cuda::CUDAStream& stream,
                                  const std::optional<std::reference_wrapper<tensor_list_t>>& tensors_to_record_opt) mutable {
            // Received token counters
            int num_recv_tokens = 0, num_expanded_tokens = 0, num_expanded_worst = 0;
            int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
            std::vector<int> num_recv_tokens_per_expert_list;

            // Assign these values according to modes
            if (cached_mode) {
                // Cached mode
                EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
                num_recv_tokens = cached_num_recv_tokens.value();
                num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
                num_expanded_tokens = cached_num_expanded_tokens.value();
            } else if (do_cpu_sync) {
                // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
                // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
                // unless we release GIL here.
                pybind11::gil_scoped_release release;

                // Non-cached mode with sync
                const auto start_cpu_time = std::chrono::high_resolution_clock::now();
                while (true) {
                    bool ready = true;

                    // Read number of received tokens from each scaleup rank
                    while (counter_scaleup_rank_idx < context->num_scaleup_ranks and ready) {
                        const auto count = math::encode_decode_positive(
                            host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);
                        if ((ready = math::is_decoded_positive_ready(count))) {
                            num_recv_tokens += count;
                            ++ counter_scaleup_rank_idx;
                        }
                    }

                    // Read expert counts
                    while (counter_local_expert_idx < num_local_experts and ready) {
                        const auto count = math::encode_decode_positive(
                            host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);
                        if ((ready = math::is_decoded_positive_ready(count))) {
                            num_recv_tokens_per_expert_list.push_back(count);
                            num_expanded_tokens += count;
                            ++ counter_local_expert_idx;
                        }
                    }

                    // Ready and do next steps
                    const auto get_buffer_info = [&]() {
                        std::stringstream ss;
                        ss << "CPU side received count (scaleup: " << context->scaleup_rank_idx << "): ";
                        for (int i = 0; i < context->num_scaleup_ranks + num_local_experts; ++ i) {
                            ss << host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i];
                            ss << (i == context->num_scaleup_ranks - 1 ? " # ": " ");
                        }
                        return ss.str();
                    };
                    if (ready) {
                        if (get_env<int>("EP_BUFFER_DEBUG"))
                            printf("%s\n", get_buffer_info().c_str());
                        break;
                    }

                    // Timeout checks
                    const auto now = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > context->num_cpu_timeout_secs)
                        throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
                }
            } else {
                // Non-cached mode without CPU sync, allocate with the worst case
                num_recv_tokens = num_max_tokens_per_rank * context->num_ranks;
                num_expanded_tokens = context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts);
                num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
                num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
                // EPArena segment: the expanded recv_x is the segment's C rows (beta x the mean), not
                // the worst case; y is leased into the same rows; rows beyond C are borrowed
                num_expanded_worst = num_expanded_tokens;
                if (seg_hd != nullptr) num_expanded_tokens = epa_seg_rows;
            }

            // Allocate received tensors
            // `recv_src_metadata` includes source token indices and buffer slot indices
            const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
            auto recv_x = out_p > 0 ? epa_out_x : seg_hd != nullptr ?
                torch::for_blob(math::advance_ptr(seg_hd, layout::EPASeg::kHeaderBytes), {num_allocated_tokens, hidden})
                    .options(x.options()).target_device(x.device()).make_tensor() :
                torch::empty({num_allocated_tokens, hidden}, x.options());
            auto recv_sf = std::optional<torch::Tensor>();
            auto recv_topk_idx = std::optional<torch::Tensor>();
            auto recv_topk_weights = std::optional<torch::Tensor>();
            auto recv_src_metadata = cached_mode ?
                cached_recv_src_metadata.value() :
                torch::empty({num_recv_tokens, num_topk + 2},
                             torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

            // Optional tensors
            void* recv_sf_ptr = nullptr;
            topk_idx_t* recv_topk_idx_ptr = nullptr;
            float* recv_topk_weights_ptr = nullptr;
            int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
            if (sf.has_value()) {
                if (not use_tma_aligned_col_major_sf) {
                    recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
                } else {
                    // TMA-aligned layout for the next GEMM input
                    recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);
                }
                recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                               {recv_sf_token_stride, recv_sf_hidden_stride},
                                               sf->options());
                recv_sf_ptr = recv_sf->data_ptr();
            }
            if (not do_expand) {
                recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
                recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
            }
            if (topk_weights.has_value()) {
                recv_topk_weights = out_p > 0 and do_expand ? epa_out_w : do_expand ?
                    torch::empty({seg_hd != nullptr ? num_expanded_worst : num_allocated_tokens}, topk_weights->options()) :
                    torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());
                recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
            }

            // Process prefix sum, in expanding mode, it is also atomic counters
            if (not cached_mode) {
                if (do_expand) {
                    // Slice the exclusive part and do atomic additions into inclusive
                    psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
                } else {
                    // Slice the inclusive part (and will not be used in the epilogue)
                    psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
                }
            }
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);

            // Launch copy kernels with full SMs
            launch_dispatch_copy_epilogue(context->buffer, context->workspace,
                                          psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                                          psum_num_recv_tokens_per_expert.data_ptr<int>(),
                                          recv_x.data_ptr(), recv_sf_ptr,
                                          recv_topk_idx_ptr, recv_topk_weights_ptr,
                                          recv_src_metadata.data_ptr<int>(),
                                          channel_linked_list_ptr,
                                          num_unaligned_recv_tokens_per_expert_ptr,
                                          num_recv_tokens, num_max_tokens_per_rank,
                                          num_hidden_bytes,
                                          num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                                          num_experts, num_topk, expert_alignment,
                                          context->scaleout_rank_idx, context->scaleup_rank_idx,
                                          context->num_scaleout_ranks, context->num_scaleup_ranks,
                                          jit->device.get_num_sms(),
                                          jit->device.get_num_smem_bytes(),
                                          num_channels,
                                          do_expand, cached_mode,
                                          do_zero_padding,
                                          stream,
                                          epa_cap, epa_pool, epa_tables, epa_peer_buffers,
                                          do_expand ? out_p : -out_p, out_pages, seg_hd != nullptr ? seg_hd : epa_out_meta);

            auto result = pybind11::make_tuple(
                recv_x, recv_sf,
                recv_topk_idx, recv_topk_weights,
                num_recv_tokens, num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                token_metadata_at_forward,
                channel_linked_list);

            // For non-deferring tensor recording
            if (tensors_to_record_opt.has_value()) {
                auto& tensors = tensors_to_record_opt->get();
                tensors.push_back(recv_x);
                tensors.push_back(recv_sf);
                tensors.push_back(recv_topk_idx);
                tensors.push_back(recv_topk_weights);
                tensors.push_back(recv_src_metadata);
            }
            return result;
        };

        // Defer epilogue: record the created tensors only, and return intermediately
        // NOTES: CPU sync will be deferred, too
        if (defer_epilogue) {
            EP_HOST_ASSERT(async_with_compute_stream);
            const auto event = stream_control_epilogue(
                tensors_to_record, compute_stream, allocate_on_comm_stream, true);
            std::function<pybind11::object()> epilogue_hook = [epilogue = std::move(epilogue)]() mutable {
                return epilogue(at::cuda::getCurrentCUDAStream(), std::nullopt);
            };
            return pybind11::make_tuple(pybind11::none(), event, epilogue_hook);
        }

        // Do epilogue intermediately and record all tensors
        auto result = epilogue(comm_stream, std::ref(tensors_to_record));
        const auto event = stream_control_epilogue(
            tensors_to_record, compute_stream, allocate_on_comm_stream, async_with_compute_stream);
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    pybind11::tuple
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const std::optional<torch::Tensor>& token_metadata_at_forward,
            const std::optional<torch::Tensor>& channel_linked_list,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms, const int& num_qps,
            const std::optional<EventHandle>& previous_event,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout_in,
            const bool& defer_epilogue,
            const std::optional<torch::Tensor>& out) const {
        const bool use_expanded_layout = use_expanded_layout_in or (epa_out_p > 0 and epa_out_normal);
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        EP_HOST_ASSERT((num_sms > 1 or context->num_scaleout_ranks == 1 or context->num_scaleup_ranks == 1) and
                       "Hybrid combine requires at least 2 SMs");
        EP_HOST_ASSERT(not defer_epilogue or async_with_compute_stream);

        // Check data
        const auto [num_tokens, hidden] = get_shape<2>(x);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

        // Check tensors at dispatch
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);
        if (out.has_value())
            EP_HOST_ASSERT(out->is_cuda() and out->is_contiguous() and out->scalar_type() == torch::kBFloat16 and
                           out->size(0) == num_combined_tokens and out->size(1) == hidden);
        const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(num_scaleup_ranks == context->num_scaleup_ranks);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

        // Check metadata
        // For reduction mode, `num_tokens_` means the number of unexpanded tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // Check optional tensors
        if (topk_weights.has_value()) {
            if (use_expanded_layout) {
                const auto [num_tokens__] = get_shape<1>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__);
            } else {
                const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
            }
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        }

        const auto bias_opts = std::vector({bias_0, bias_1});
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_cuda() and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream);
        const auto comm_stream = comm::get_comm_stream();

        // Check buffer size
        EP_HOST_ASSERT(get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                                               context->num_scaleout_ranks, context->num_scaleup_ranks,
                                               context->is_scaleup_nvlink, allow_multiple_reduction) <= num_buffer_bytes);

        // Optional configs and metadata for hybrid combine
        int num_channels = 1;
        int* token_metadata_at_forward_ptr = nullptr;
        int* channel_linked_list_ptr = nullptr;
        if (context->num_scaleout_ranks > 1) {
            // The token metadata during forward
            const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward.value());
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
            num_channels = num_channels_;
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();
            EP_HOST_ASSERT(d1 == context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
            EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
            EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);

            // Per-scaleup-peer-per-channel linked list
            const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
            EP_HOST_ASSERT(num_channels == num_channels__);
            EP_HOST_ASSERT(d1_ == context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2_ == context->num_scaleup_ranks);
            EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
            EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
        }

        // Push data into remote buffers
        // NOTES: we don't use `num_hidden_bytes` due to enable later quantization possibility
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            context->dev_comm, context->window,
            context->buffer, context->workspace,
            num_reduced_tokens, num_max_tokens_per_rank,
            hidden, num_experts, num_topk,
            num_qps, context->num_gpu_timeout_cycles,
            context->num_scaleout_ranks, context->num_scaleup_ranks,
            context->scaleout_rank_idx, context->scaleup_rank_idx,
            context->is_scaleup_nvlink,
            num_sms, jit->device.get_num_smem_bytes(),
            num_channels,
            use_expanded_layout, allow_multiple_reduction,
            comm_stream
            // EPArena segment: x is the segment's rows (y leased into recv_x): rows >= C are borrowed
            , (epa_seg_hd != nullptr and x.data_ptr() == math::advance_ptr(epa_seg_hd, layout::EPASeg::kHeaderBytes)) ? epa_seg_hd : nullptr);

        // For tensor recording
        tensor_list_t tensors_to_record = {
            x, topk_weights, bias_0, bias_1,
            src_metadata, combined_topk_idx,
            psum_num_recv_tokens_per_scaleup_rank,
            token_metadata_at_forward, channel_linked_list};

        // Epilogue can be deferred, so it is a lambda
        auto epilogue = [=, this](const at::cuda::CUDAStream& stream,
                                  const std::optional<std::reference_wrapper<tensor_list_t>>& tensors_to_record_opt) {
            // Allocate output tensors
            // `out` (optional): the caller's buffer for the result, e.g. memory that is dead by now
            auto combined_x = out.has_value() ? *out : torch::empty({num_combined_tokens, hidden}, x.options());
            auto combined_topk_weights = std::optional<torch::Tensor>();
            float* combined_topk_weights_ptr = nullptr;
            if (topk_weights.has_value()) {
                combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
                combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
            }

            // Resolve pointers here to retain bias tensors in a deferred epilogue
            void* bias_ptrs[2] = {
                bias_opts[0].has_value() ? bias_opts[0]->data_ptr() : nullptr,
                bias_opts[1].has_value() ? bias_opts[1]->data_ptr() : nullptr
            };

            // Combine pushed data
            launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                           combined_topk_weights_ptr,
                                           combined_topk_idx.data_ptr<topk_idx_t>(),
                                           num_combined_tokens, num_max_tokens_per_rank,
                                           hidden,
                                           num_experts, num_topk,
                                           reduce_buffer,
                                           bias_ptrs[0], bias_ptrs[1],
                                           context->num_scaleout_ranks, context->num_scaleup_ranks,
                                           context->scaleout_rank_idx, context->scaleup_rank_idx,
                                           jit->device.get_num_sms(),
                                           jit->device.get_num_smem_bytes(),
                                           use_expanded_layout, allow_multiple_reduction,
                                           stream);

            if (tensors_to_record_opt.has_value()) {
                auto& tensors = tensors_to_record_opt->get();
                tensors.push_back(combined_x);
                tensors.push_back(combined_topk_weights);
            }
            return pybind11::make_tuple(combined_x, combined_topk_weights);
        };

        // Defer epilogue
        if (defer_epilogue) {
            const auto event = stream_control_epilogue(
                tensors_to_record, compute_stream, allocate_on_comm_stream, true);
            std::function<pybind11::object()> epilogue_hook = [epilogue = std::move(epilogue)]() {
                return epilogue(at::cuda::getCurrentCUDAStream(), std::nullopt);
            };
            return pybind11::make_tuple(pybind11::none(), event, epilogue_hook);
        }

        // Do epilogue
        auto result = epilogue(comm_stream, std::ref(tensors_to_record));
        const auto event = stream_control_epilogue(
            tensors_to_record, compute_stream, allocate_on_comm_stream, async_with_compute_stream);
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    std::optional<EventHandle>
    lb_prefetch_weights(const std::vector<torch::Tensor>& redundant_expert_weights,
                        const std::vector<torch::Tensor>& expert_weights,
                        const torch::Tensor& redundancy_mapping,
                        const int& num_sms,
                        const std::optional<EventHandle>& previous_event) const {
        // Checks
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        EP_HOST_ASSERT(redundant_expert_weights.size() == expert_weights.size());
        EP_HOST_ASSERT(not expert_weights.empty());
        EP_HOST_ASSERT(redundant_expert_weights.size() <= layout::kNumMaxWeightEntries);
        EP_HOST_ASSERT(redundancy_mapping.is_cuda() and redundancy_mapping.is_contiguous());
        EP_HOST_ASSERT(redundancy_mapping.dim() == 2);
        EP_HOST_ASSERT(redundancy_mapping.size(0) == context->num_nvl_ranks);
        EP_HOST_ASSERT(redundancy_mapping.scalar_type() == torch::kInt);

        const int num_redundant_experts = redundancy_mapping.size(1);
        const int num_weights = static_cast<int>(expert_weights.size());
        const int num_local_experts = expert_weights.front().size(0);
        EP_HOST_ASSERT(num_local_experts > 0);

        layout::EPWeightList weights;
        for (int i = 0; i < num_weights; ++ i) {
            const auto& redundant_expert_weight = redundant_expert_weights[i];
            const auto& expert_weight = expert_weights[i];
            EP_HOST_ASSERT(redundant_expert_weight.is_cuda() and redundant_expert_weight.is_contiguous());
            EP_HOST_ASSERT(expert_weight.is_cuda() and expert_weight.is_contiguous());
            EP_HOST_ASSERT(redundant_expert_weight.dim() >= 1);
            EP_HOST_ASSERT(expert_weight.dim() >= 1);
            EP_HOST_ASSERT(redundant_expert_weight.size(0) == num_redundant_experts);
            EP_HOST_ASSERT(expert_weight.size(0) == num_local_experts);
            const int64_t num_bytes_per_expert = c10::multiply_integers(redundant_expert_weight.sizes().slice(1)) *
                                                  redundant_expert_weight.element_size();
            EP_HOST_ASSERT(num_bytes_per_expert == c10::multiply_integers(expert_weight.sizes().slice(1)) *
                                                   expert_weight.element_size());
            EP_HOST_ASSERT(num_bytes_per_expert % 16 == 0);

            weights[i] = {
                .redundant_expert_weights = redundant_expert_weight.data_ptr(),
                .expert_weights = expert_weight.data_ptr(),
                .num_bytes_per_expert = num_bytes_per_expert
            };
        }

        // Stream control
        const auto compute_stream = stream_control_prologue(previous_event);

        // Launch
        launch_lb_prefetch_weights(
            *context, num_weights, weights, redundancy_mapping.data_ptr<int>(),
            num_redundant_experts, num_local_experts,
            num_sms, comm::get_comm_stream());

        // Stream epilogue
        tensor_list_t tensors_to_record = {redundancy_mapping};
        tensors_to_record.insert(tensors_to_record.end(), redundant_expert_weights.begin(), redundant_expert_weights.end());
        tensors_to_record.insert(tensors_to_record.end(), expert_weights.begin(), expert_weights.end());
        return stream_control_epilogue(tensors_to_record, compute_stream, false, true);
    }

    std::optional<EventHandle>
    lb_reduce_grads(const torch::Tensor& redundant_expert_grads,
                    const torch::Tensor& expert_grads,
                    const torch::Tensor& redundancy_mapping,
                    const int& num_sms,
                    const std::optional<EventHandle>& previous_event) const {
        // Checks
        EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
        EP_HOST_ASSERT(redundant_expert_grads.is_cuda() and redundant_expert_grads.is_contiguous());
        EP_HOST_ASSERT(expert_grads.is_cuda() and expert_grads.is_contiguous());
        EP_HOST_ASSERT(redundancy_mapping.is_cuda() and redundancy_mapping.is_contiguous());
        EP_HOST_ASSERT(redundant_expert_grads.dim() == 2);
        EP_HOST_ASSERT(expert_grads.dim() == 2);
        EP_HOST_ASSERT(redundancy_mapping.dim() == 2);
        EP_HOST_ASSERT(redundancy_mapping.size(0) == context->num_nvl_ranks);
        EP_HOST_ASSERT(redundancy_mapping.scalar_type() == torch::kInt);

        const int num_redundant_experts = redundancy_mapping.size(1);
        const auto [num_local_experts, hidden] = get_shape<2>(expert_grads);

        EP_HOST_ASSERT(redundant_expert_grads.size(0) == num_redundant_experts);
        EP_HOST_ASSERT(redundant_expert_grads.size(1) == hidden);
        EP_HOST_ASSERT(redundant_expert_grads.scalar_type() == torch::kFloat);
        EP_HOST_ASSERT(expert_grads.scalar_type() == torch::kFloat);

        // Stream control
        const auto compute_stream = stream_control_prologue(previous_event);

        // Launch: accumulate peers' redundant gradients into the local expert gradients
        launch_lb_reduce_grads(
            *context, redundant_expert_grads.data_ptr<float>(), expert_grads.data_ptr<float>(),
            redundancy_mapping.data_ptr<int>(), num_redundant_experts, num_local_experts, hidden,
            num_sms, comm::get_comm_stream());

        // Stream epilogue
        return stream_control_epilogue(
            {redundant_expert_grads, expert_grads, redundancy_mapping},
            compute_stream, false, true);
    }
    // EPArena (arena mode), for measurement: this rank's per-destination take state of the
    // last dispatch, which outlives the call (`taken` and `avail` are cleared at its end):
    // rows [have, got, pred]; pages held on destination d = pred > have ? have + got : pred
    // Paged expanded output, for the expert and for measurement:
    // (y, recv_x, P, pages, stats [held, failed takes, dropped rows, max held]); tile_expert via epa_out_tile_expert
    pybind11::tuple epa_out_state() const {
        EP_HOST_ASSERT(epa_out_meta != nullptr and "no paged dispatch yet");
        const auto opts = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
        return pybind11::make_tuple(epa_out_y, epa_out_x, epa_out_p, epa_out_pages,
                                    torch::from_blob(epa_out_meta, {16}, opts));
    }

    // Dense valid-tile list of the last paged dispatch: [count, page of tile 0, page of tile 1, ...]
    torch::Tensor epa_out_tiles(const int& num_local_experts) const {
        EP_HOST_ASSERT(epa_out_meta != nullptr and "no paged dispatch yet");
        const auto out = layout::EPAOutLayout{epa_out_meta, epa_out_keys, epa_out_pages, epa_out_p};
        return torch::from_blob(out.tiles(), {2 * (epa_out_pages + 1)}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    }

    // EPArena segment: initialise the header of C rows of row_bytes (call on every rank, then barrier)
    void epa_seg_init(const int& rows, const int64_t& row_bytes) const { epa_seg_setup(rows, row_bytes); CUDA_RUNTIME_CHECK(cudaDeviceSynchronize()); }
    // The header as int32 [1024] (chunks at [16 + 4j], segment row addresses as int64 at [128 + 2r])
    torch::Tensor epa_seg_header() const {
        EP_HOST_ASSERT(epa_seg_hd != nullptr and "no EPArena segment yet");
        return torch::from_blob(epa_seg_hd, {layout::EPASeg::kHeaderBytes / 4}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    }
    torch::Tensor epa_out_tile_expert(const int& num_local_experts) const {
        EP_HOST_ASSERT(epa_out_meta != nullptr and "no paged dispatch yet");
        const auto out = layout::EPAOutLayout{epa_out_meta, epa_out_keys, epa_out_pages, epa_out_p};
        return torch::from_blob(out.tile_expert(), {epa_out_pages}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    }

    // EPArena (arena mode): receive pages this rank holds for each source rank (sticky, receiver-managed)
    torch::Tensor epa_recv_held(const int& num_max_tokens_per_rank) const {
        const auto [epa_cap, epa_pool] = get_epa_config(context->num_scaleout_ranks, context->is_scaleup_nvlink);
        EP_HOST_ASSERT(epa_cap < -1 and "not in arena mode");
        const int R = context->num_ranks;
        const int64_t tables = num_buffer_bytes - layout::EPALayout::num_table_bytes(R, epa_cap, epa_pool, num_max_tokens_per_rank);
        const auto epa = layout::EPALayout(0, R, num_max_tokens_per_rank, epa_cap, epa_pool, tables);
        return torch::from_blob(epa.held(context->buffer), {R}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    }

    torch::Tensor epa_take_state(const int& num_max_tokens_per_rank) const {
        const auto [epa_cap, epa_pool] = get_epa_config(context->num_scaleout_ranks, context->is_scaleup_nvlink);
        EP_HOST_ASSERT(epa_cap < -1 and "not in arena mode");
        const int R = context->num_ranks;
        const int64_t tables = num_buffer_bytes - layout::EPALayout::num_table_bytes(R, epa_cap, epa_pool, num_max_tokens_per_rank);
        const auto epa = layout::EPALayout(0, R, num_max_tokens_per_rank, epa_cap, epa_pool, tables);
        const auto opts = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
        const auto scratch = torch::from_blob(epa.scratch_ints(context->buffer), {4, R}, opts);   // have, want, first, got
        const auto pred = torch::from_blob(epa.pred(context->buffer), {1, R}, opts);
        return torch::cat({scratch.narrow(0, 0, 1), scratch.narrow(0, 3, 1), pred});
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<EPBuffer, BufferBase>(m, "EPBuffer")
        .def(pybind11::init<int, int, int64_t, int64_t, int64_t, bool, bool, bool, std::optional<int>, int, int, int, bool>())
        .def_readonly("context", &EPBuffer::context)
        .def_readonly("lb_storage", &EPBuffer::lb_storage)
        .def("dispatch", &EPBuffer::dispatch)
        .def("combine", &EPBuffer::combine)
        .def("epa_take_state", &EPBuffer::epa_take_state)
        .def("epa_out_state", &EPBuffer::epa_out_state)
        .def("epa_out_tile_expert", &EPBuffer::epa_out_tile_expert)
        .def("epa_seg_init", &EPBuffer::epa_seg_init)
        .def("epa_seg_header", &EPBuffer::epa_seg_header)
        .def("epa_out_tiles", &EPBuffer::epa_out_tiles)
        .def("epa_recv_held", &EPBuffer::epa_recv_held)
        .def("lb_prefetch_weights", &EPBuffer::lb_prefetch_weights)
        .def("lb_reduce_grads", &EPBuffer::lb_reduce_grads);
    m.def("calculate_ep_buffer_size", &EPBuffer::calculate_buffer_size);
    m.def("get_ep_buffer_alignment", [=]() {
        return kNumAllocationAlignmentBytes;
    });

}

}  // namespace deep_ep::ep
