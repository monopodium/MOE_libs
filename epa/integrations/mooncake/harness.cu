// Standalone NVLink harness for Mooncake EP's legacy low-latency kernels.
//
// It runs the unmodified upstream kernels (`mooncake::dispatch/combine`) and the
// EPArena versions (`mooncake_epa::`) on buffers that this file lays out in an
// EPArena symmetric heap (cudaMalloc + cudaIpc). Mooncake's TransferEngine is not
// needed on a single NVLink node: every peer is reached through `ipc_peer_ptrs`,
// exactly as Mooncake's own P2P transport does.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <mooncake_ep_combine.cuh>
#include <mooncake_ep_dispatch.cuh>
#include <mooncake_ep_launch.cuh>
#include "epa_combine.cuh"
#include "epa_pull.cuh"
#ifdef EPA_TRACE
#include "epa_dispatch_trace.cuh"
#else
#include "epa_dispatch.cuh"
#include "arena_dispatch.cuh"
#endif

#include <epa/epa.cuh>
#include <epa/heap.hpp>

#include <map>
#include <memory>

namespace {

constexpr int kWarpsPerGroup = 4, kWarpGroups = 8, kMaxTopk = 17;

struct Inst {
    bool epa = false, fp8 = false, arena = false;   // arena: dispatch rows in epa::Arena pages
    int T = 0, H = 0, E = 0, K = 0, R = 0, rank = 0, cap = 0, pool = 0;
    epa::Window win;   // dispatch receive window: keys (local expert, source), keyed = cap, shared = pool
    epa::Arena ar;     // arena mode: rows per page = cap, pages for `pool` rows + one per key
    int rows_per_page = 0, max_pages = 0;
    int64_t ar_off = 0, status[2];
    int* table_local = nullptr;
    int* avail_local = nullptr;   // [E] pages usable now (sender)
    int* pred_local = nullptr;    // [E] pages each key took last call (sender)
    std::unique_ptr<epa::SymmetricHeap> heap;
    int64_t sig_send[2], sig_recv[2], send_data[2], recv_data[2], recv_comb[2], desc[2], tbl[2];
    int64_t total = 0;
    int* workspace = nullptr;
    int32_t* nvlink_avail = nullptr;
    int32_t* active_ranks = nullptr;
    int buffer_idx = 0;

    template <typename T_ = void> T_* at(int64_t off) const {
        return reinterpret_cast<T_*>(static_cast<char*>(heap->base) + off);
    }
};

std::map<int64_t, Inst> g_insts;
int64_t g_next = 0;

int64_t msg_bytes(int H, bool fp8) { return 16 + (fp8 ? H + H / 128 * 4 : H * 2); }

// Layout the two buffers. Upstream: Mooncake's BufferPair (4 signal + 4 data regions,
// each data region E*T*(32 + 2H) bytes). EPA: signals + home + max(tier-1 slots +
// descriptors, combine [K][T] slots), planned by phase with epa::plan_regions.
void layout(Inst& s) {
    const int64_t E = s.E, T = s.T, H = s.H;
    if (not s.epa) {
        const int64_t sig = E * 4, data = E * T * (2 * 16 + H * 2);
        for (int i = 0; i < 2; ++i) {
            const int64_t base = i * (2 * sig + 2 * data);
            s.sig_send[i] = base, s.sig_recv[i] = base + sig;
            s.send_data[i] = base + 2 * sig, s.recv_data[i] = base + 2 * sig + data;
            s.desc[i] = s.tbl[i] = 0, s.recv_comb[i] = s.recv_data[i];
        }
        s.total = 2 * (2 * sig + 2 * data);
        return;
    }
    const int64_t msg = 16 + H * 2;   // home row sized for BF16 (FP8 rows are smaller)
    // Phases are the joint states of the two buffers: bit (2 * t0 + t1) with t = 0
    // for dispatch, 1 for combine. Buffer i's dispatch slots and combine slots are
    // never live together, so the planner overlaps them; anything on buffer 0 can
    // be live with anything on buffer 1, so those never overlap.
    auto live = [](int buf, int type) {
        uint32_t m = 0;
        for (int t0 = 0; t0 < 2; ++t0)
            for (int t1 = 0; t1 < 2; ++t1)
                if ((buf == 0 ? t0 : t1) == type) m |= 1u << (2 * t0 + t1);
        return m;
    };
    std::vector<epa::RegionSpec> specs;
    for (int i = 0; i < 2; ++i) {
        const auto p = std::to_string(i);
        specs.push_back({"sig_send" + p, E * 4, 0xf});
        specs.push_back({"sig_recv" + p, (E + 1) * 4, 0xf});   // + tier-2 pool head
        specs.push_back({"home" + p, T * msg, 0xf});
        // dispatch receive: the window (keyed + shared rows), descriptors of rows left
        // in Home, and the per-key shared base table. Arena mode: page table, descriptors
        // and status per key; the rows live in the arena shared by both buffers.
        const int64_t rows_bytes = s.arena ? E * s.max_pages * 4 : s.win.rows() * msg;
        specs.push_back({"slots" + p, rows_bytes + E * T * 4 + E * 4, live(i, 0)});
        specs.push_back({"comb" + p, static_cast<int64_t>(s.K) * T * H * 2, live(i, 1)});
    }
    if (s.arena)   // ring state persists across calls: live in every phase
        specs.push_back({"arena", epa::Arena::bytes_for(s.ar.num_pages, s.ar.page_bytes), 0xf});
    auto planned = epa::plan_regions(specs, &s.total);
    auto off = [&](const std::string& n) {
        for (auto& p : planned) if (p.name == n) return p.offset;
        throw std::runtime_error("no region " + n);
    };
    for (int i = 0; i < 2; ++i) {
        const auto p = std::to_string(i);
        s.sig_send[i] = off("sig_send" + p), s.sig_recv[i] = off("sig_recv" + p);
        s.send_data[i] = off("home" + p);
        s.recv_data[i] = off("slots" + p);
        s.recv_comb[i] = off("comb" + p);
        s.desc[i] = s.recv_data[i] + (s.arena ? E * s.max_pages * 4 : s.win.rows() * msg);
        s.tbl[i] = s.desc[i] + E * T * 4;
        s.status[i] = s.tbl[i];
    }
    if (s.arena) s.ar_off = off("arena");
}

std::vector<int64_t> create(int64_t mode, int64_t T, int64_t H, int64_t E, int64_t K, int64_t cap,
                            int64_t pool, bool fp8, int64_t rank, int64_t R) {
    Inst s;
    s.epa = mode >= 1, s.arena = mode == 2, s.fp8 = fp8, s.T = T, s.H = H, s.E = E, s.K = K, s.cap = cap, s.pool = pool;
    s.rank = rank, s.R = R;
    s.win.num_keys = E, s.win.keyed = cap, s.win.shared = pool, s.win.max_rows = E * T;
    if (s.arena) {
        s.rows_per_page = static_cast<int>(cap);
        s.max_pages = static_cast<int>((T + cap - 1) / cap);
        s.ar.page_bytes = cap * (16 + H * 2);   // BF16 messages (FP8 ones are smaller)
        s.ar.num_pages = static_cast<int>((pool + cap - 1) / cap + E);   // + one partial page per key
    }
    layout(s);
    s.heap = std::make_unique<epa::SymmetricHeap>(s.total);
    if (s.arena) {
        s.ar.base = s.at<char>(s.ar_off);
        epa::arena_init_kernel<<<64, 256>>>(s.ar);
        EPA_CUDA_CHECK(cudaMalloc(&s.table_local, E * s.max_pages * 4));
        EPA_CUDA_CHECK(cudaMalloc(&s.avail_local, E * 4));
        EPA_CUDA_CHECK(cudaMalloc(&s.pred_local, E * 4));
        EPA_CUDA_CHECK(cudaMemset(s.avail_local, 0, E * 4));
        EPA_CUDA_CHECK(cudaMemset(s.pred_local, 0, E * 4));
        EPA_CUDA_CHECK(cudaDeviceSynchronize());
    }
    EPA_CUDA_CHECK(cudaMalloc(&s.workspace, NUM_WORKSPACE_BYTES));
    EPA_CUDA_CHECK(cudaMemset(s.workspace, 0, NUM_WORKSPACE_BYTES));
    EPA_CUDA_CHECK(cudaMalloc(&s.nvlink_avail, R * 4));
    EPA_CUDA_CHECK(cudaMalloc(&s.active_ranks, R * 4));
    std::vector<int32_t> ones(R, 1);
    EPA_CUDA_CHECK(cudaMemcpy(s.nvlink_avail, ones.data(), R * 4, cudaMemcpyHostToDevice));
    EPA_CUDA_CHECK(cudaMemcpy(s.active_ranks, ones.data(), R * 4, cudaMemcpyHostToDevice));
    const auto id = g_next++;
    const auto total = s.total;
    g_insts.emplace(id, std::move(s));
    return {id, total};
}

torch::Tensor handle(int64_t id) {
    auto v = g_insts.at(id).heap->ipc_handle();
    auto t = torch::empty({static_cast<int64_t>(v.size())}, torch::kUInt8);
    std::memcpy(t.data_ptr(), v.data(), v.size());
    return t;
}

void open_peers(int64_t id, torch::Tensor all) {
    std::vector<std::vector<uint8_t>> hs(all.size(0));
    for (int64_t r = 0; r < all.size(0); ++r)
        hs[r].assign(all[r].data_ptr<uint8_t>(), all[r].data_ptr<uint8_t>() + all.size(1));
    auto& s = g_insts.at(id);
    s.heap->open_peers(hs, s.rank);
}

// [tier-2 messages, tier-3 messages] received since the last call; resets them
std::vector<int64_t> stats(int64_t id) {
    auto& s = g_insts.at(id);
    int v[2] = {0, 0};
    if (s.epa) {
        EPA_CUDA_CHECK(cudaMemcpy(v, s.workspace + 5 * s.E, 8, cudaMemcpyDeviceToHost));
        EPA_CUDA_CHECK(cudaMemset(s.workspace + 5 * s.E, 0, 8));
    }
    return {v[0], v[1]};
}

// Arena mode: {pages, peak pages in use, page bytes, failed takes}
std::vector<int64_t> arena_stats(int64_t id) {
    auto& s = g_insts.at(id);
    if (not s.arena) return {0, 0, 0, 0};
    epa::ArenaMeta m;
    EPA_CUDA_CHECK(cudaMemcpy(&m, s.ar.base, sizeof m, cudaMemcpyDeviceToHost));
    return {s.ar.num_pages, s.ar.num_pages - m.min_free_pages, s.ar.page_bytes, static_cast<int64_t>(m.failed)};
}

#ifdef EPA_TRACE
torch::Tensor trace() {
    auto t = torch::empty({64, 8}, torch::kInt64);
    EPA_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), epa_trace, sizeof(unsigned long long) * 64 * 8));
    return t;
}
#else
torch::Tensor trace() { return torch::empty({0}); }
#endif

void destroy(int64_t id) {
    auto& s = g_insts.at(id);
    cudaFree(s.workspace), cudaFree(s.nvlink_avail), cudaFree(s.active_ranks);
    if (s.arena) cudaFree(s.table_local), cudaFree(s.avail_local), cudaFree(s.pred_local);
    g_insts.erase(id);
}

template <typename Args>
Args dispatch_args(Inst& s, int cur, int nxt, const torch::Tensor& x, const torch::Tensor& topk_idx,
                   const torch::Tensor& recv_x, const std::optional<torch::Tensor>& recv_scales,
                   const torch::Tensor& src_info, const torch::Tensor& layout_range,
                   const torch::Tensor& recv_count) {
    Args a{};
    a.packed_recv_x = recv_x.data_ptr();
    a.packed_recv_x_scales = recv_scales ? recv_scales->data_ptr<float>() : nullptr;
    a.packed_recv_src_info = src_info.data_ptr<int>();
    a.packed_recv_layout_range = layout_range.data_ptr<int64_t>();
    a.packed_recv_count = recv_count.data_ptr<int>();
    a.cumulative_local_expert_recv_stats = nullptr;
    a.active_ranks = s.active_ranks;
    a.mxa_buffer = s.heap->base;
    a.rdma_send_signal_buffer = s.at<int>(s.sig_send[cur]);
    a.rdma_recv_signal_buffer = s.at<int>(s.sig_recv[cur]);
    a.rdma_send_data_buffer = s.at(s.send_data[cur]);
    a.rdma_recv_data_buffer = s.at(s.recv_data[cur]);
    a.nvlink_available = s.nvlink_avail;
    a.ipc_peer_ptrs = s.heap->d_peer_bases;
    a.x = x.data_ptr();
    a.topk_idx = topk_idx.data_ptr<int64_t>();
    a.atomic_counter_per_expert = s.workspace;
    a.atomic_finish_counter_per_expert = s.workspace + s.E;
    a.next_clean_buffer = s.at<int>(s.sig_recv[nxt]);
    a.num_tokens = x.size(0);
    a.num_max_dispatch_tokens_per_rank = s.T;
    a.num_topk = topk_idx.size(1);
    a.num_experts = s.E;
    a.rank = s.rank;
    a.num_ranks = s.R;
    a.qp_num_ranks = s.R;
    a.timeout_ticks = -1;
    a.phases = LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE;
    a.active_qps_per_rank = 1;
    return a;
}

void dispatch(int64_t id, torch::Tensor x, torch::Tensor topk_idx, torch::Tensor recv_x,
              std::optional<torch::Tensor> recv_scales, torch::Tensor src_info,
              torch::Tensor layout_range, torch::Tensor recv_count) {
    auto& s = g_insts.at(id);
    const int cur = s.buffer_idx, nxt = s.buffer_idx ^= 1;
    const int num_sms = std::max(2, (s.E + kWarpGroups - 1) / kWarpGroups);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    SETUP_LAUNCH_CONFIG(num_sms, kWarpGroups * kWarpsPerGroup * 32, stream);
    EPA_CUDA_CHECK(s.H == 7168 ? cudaSuccess : cudaErrorInvalidValue);
    if (s.arena) {
        auto a = dispatch_args<mooncake_arena::DispatchKernelArgs>(s, cur, nxt, x, topk_idx, recv_x, recv_scales,
                                                                   src_info, layout_range, recv_count);
        a.epa_arena = s.ar;
        a.epa_rows_per_page = s.rows_per_page;
        a.epa_max_pages = s.max_pages;
        a.epa_table = s.at<int>(s.recv_data[cur]);
        a.epa_table_local = s.table_local;
        a.epa_taken = s.at<int>(s.status[cur]);
        a.epa_avail_local = s.avail_local;
        a.epa_pred_local = s.pred_local;
        a.epa_recv_desc = s.at<int>(s.desc[cur]);
        a.epa_stats = s.workspace + 5 * s.E;
        auto k = s.fp8 ? mooncake_arena::dispatch<true, kWarpGroups, kWarpsPerGroup, 7168>
                       : mooncake_arena::dispatch<false, kWarpGroups, kWarpsPerGroup, 7168>;
        LAUNCH_KERNEL(&cfg, k, a);
    } else if (s.epa) {
        auto a = dispatch_args<mooncake_epa::DispatchKernelArgs>(s, cur, nxt, x, topk_idx, recv_x, recv_scales,
                                                                 src_info, layout_range, recv_count);
        a.epa_win = s.win;
        a.epa_recv_desc = s.at<int>(s.desc[cur]);
        a.epa_pool_head = s.at<int>(s.sig_recv[cur]) + s.E;
        a.epa_pool_tbl = s.at<int>(s.tbl[cur]);
        a.epa_pool_base_local = s.workspace + 4 * s.E;
        a.epa_stats = s.workspace + 5 * s.E;
        auto k = s.fp8 ? mooncake_epa::dispatch<true, kWarpGroups, kWarpsPerGroup, 7168>
                       : mooncake_epa::dispatch<false, kWarpGroups, kWarpsPerGroup, 7168>;
        LAUNCH_KERNEL(&cfg, k, a);
    } else {
        auto a = dispatch_args<mooncake::DispatchKernelArgs>(s, cur, nxt, x, topk_idx, recv_x, recv_scales,
                                                             src_info, layout_range, recv_count);
        auto k = s.fp8 ? mooncake::dispatch<true, kWarpGroups, kWarpsPerGroup, 7168>
                       : mooncake::dispatch<false, kWarpGroups, kWarpsPerGroup, 7168>;
        LAUNCH_KERNEL(&cfg, k, a);
    }
}

template <typename Args>
Args combine_args(Inst& s, int cur, int nxt, const torch::Tensor& x, const torch::Tensor& topk_idx,
                  const torch::Tensor& topk_weights, const torch::Tensor& src_info,
                  const torch::Tensor& layout_range, const torch::Tensor& combined_x) {
    Args a{};
    a.combined_x = combined_x.data_ptr();
    a.active_ranks = s.active_ranks;
    a.mxa_buffer = s.heap->base;
    a.rdma_send_signal_buffer = s.at<int>(s.sig_send[cur]);
    a.rdma_recv_signal_buffer = s.at<int>(s.sig_recv[cur]);
    a.rdma_send_data_buffer = s.at(s.send_data[cur]);
    a.rdma_recv_data_buffer = s.at(s.recv_comb[cur]);
    a.nvlink_available = s.nvlink_avail;
    a.ipc_peer_ptrs = s.heap->d_peer_bases;
    a.x = x.data_ptr();
    a.topk_idx = topk_idx.data_ptr<int64_t>();
    a.topk_weights = topk_weights.data_ptr<float>();
    a.src_info = src_info.data_ptr<int>();
    a.layout_range = layout_range.data_ptr<int64_t>();
    a.next_clean_buffer = s.at<int>(s.sig_recv[nxt]);
    a.atomic_clean_flag = s.workspace;
    a.num_combined_tokens = combined_x.size(0);
    a.num_topk = topk_idx.size(1);
    a.num_max_dispatch_tokens_per_rank = s.T;
    a.num_experts = s.E;
    a.rank = s.rank;
    a.num_ranks = s.R;
    a.qp_num_ranks = s.R;
    a.timeout_ticks = -1;
    a.phases = LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE;
    a.zero_copy = false;
    a.active_qps_per_rank = 1;
    return a;
}

void combine(int64_t id, torch::Tensor x, torch::Tensor topk_idx, torch::Tensor topk_weights,
             torch::Tensor src_info, torch::Tensor layout_range, torch::Tensor combined_x) {
    auto& s = g_insts.at(id);
    const int cur = s.buffer_idx, nxt = s.buffer_idx ^= 1;
    const int num_sms = (s.E + kWarpGroups - 1) / kWarpGroups;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    SETUP_LAUNCH_CONFIG(num_sms, kWarpGroups * kWarpsPerGroup * 32, stream);
    if (s.epa) {
        auto a = combine_args<mooncake_epa::CombineKernelArgs>(s, cur, nxt, x, topk_idx, topk_weights, src_info,
                                                               layout_range, combined_x);
        LAUNCH_KERNEL(&cfg, (mooncake_epa::combine<kWarpGroups, kWarpsPerGroup, 7168, kMaxTopk>), a);
    } else {
        auto a = combine_args<mooncake::CombineKernelArgs>(s, cur, nxt, x, topk_idx, topk_weights, src_info,
                                                           layout_range, combined_x);
        LAUNCH_KERNEL(&cfg, (mooncake::combine<kWarpGroups, kWarpsPerGroup, 7168, kMaxTopk>), a);
    }
}

// ------------------------------------------------------------------ EPA-pull
struct PullInst {
    epa_pull::Layout l;
    std::unique_ptr<epa::SymmetricHeap> heap;
    int* work = nullptr;
    int T, H, E, K, R, rank, num_blocks;
    bool fp8;
};
std::map<int64_t, PullInst> g_pull;

std::vector<int64_t> pull_create(int64_t T, int64_t H, int64_t E, int64_t K, bool fp8, int64_t rank, int64_t R,
                                 int64_t num_blocks) {
    PullInst p;
    p.l = epa_pull::Layout::make(T, H, E, K, R);
    p.T = T, p.H = H, p.E = E, p.K = K, p.R = R, p.rank = rank, p.fp8 = fp8;
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    p.num_blocks = num_blocks > 0 ? num_blocks : sms;
    p.heap = std::make_unique<epa::SymmetricHeap>(p.l.total);
    EPA_CUDA_CHECK(cudaMalloc(&p.work, 3 * E * 4));
    EPA_CUDA_CHECK(cudaMemset(p.work, 0, 3 * E * 4));
    const auto id = 1000000 + g_next++;
    const int64_t comm = p.l.comm_bytes(), total = p.l.total;
    g_pull.emplace(id, std::move(p));
    return {id, comm, total};
}

torch::Tensor pull_handle(int64_t id) {
    auto v = g_pull.at(id).heap->ipc_handle();
    auto t = torch::empty({static_cast<int64_t>(v.size())}, torch::kUInt8);
    std::memcpy(t.data_ptr(), v.data(), v.size());
    return t;
}

void pull_open(int64_t id, torch::Tensor all) {
    std::vector<std::vector<uint8_t>> hs(all.size(0));
    for (int64_t r = 0; r < all.size(0); ++r)
        hs[r].assign(all[r].data_ptr<uint8_t>(), all[r].data_ptr<uint8_t>() + all.size(1));
    auto& p = g_pull.at(id);
    p.heap->open_peers(hs, p.rank);
}

// The expert output y lives in the heap: the GEMM writes it here and peers pull it
torch::Tensor pull_y(int64_t id) {
    auto& p = g_pull.at(id);
    return torch::from_blob(static_cast<char*>(p.heap->base) + p.l.y, {p.E / p.R, int64_t(p.R) * p.T, p.H},
                            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
}

void pull_destroy(int64_t id) {
    cudaFree(g_pull.at(id).work);
    g_pull.erase(id);
}

epa_pull::Args pull_args(PullInst& p) {
    epa_pull::Args a{};
    a.pt = {p.heap->base, p.heap->d_peer_bases};
    a.l = p.l;
    a.work = p.work;
    a.T = p.T, a.E = p.E, a.K = p.K, a.R = p.R, a.rank = p.rank;
    // the GEMM input is the window: one key per local expert, R*T rows each, one round
    a.win.num_keys = p.E / p.R, a.win.keyed = int64_t(p.R) * p.T, a.win.shared = 0;
    a.win.max_rows = int64_t(p.R) * p.T;
    return a;
}

void launch_coop(const void* fn, int blocks, epa_pull::Args& a) {
    void* params[] = {&a};
    EPA_CUDA_CHECK(cudaLaunchCooperativeKernel(fn, blocks, 512, params, 0, at::cuda::getCurrentCUDAStream().stream()));
}

void pull_dispatch(int64_t id, torch::Tensor x, torch::Tensor topk_idx, torch::Tensor recv_x,
                   std::optional<torch::Tensor> recv_scales, torch::Tensor src_info,
                   torch::Tensor layout_range, torch::Tensor recv_count) {
    auto& p = g_pull.at(id);
    auto a = pull_args(p);
    a.x = reinterpret_cast<const nv_bfloat16*>(x.data_ptr());
    a.topk_idx = topk_idx.data_ptr<int64_t>();
    a.packed_recv_x = recv_x.data_ptr();
    a.packed_recv_x_scales = recv_scales ? recv_scales->data_ptr<float>() : nullptr;
    a.packed_recv_src_info = src_info.data_ptr<int>();
    a.packed_recv_layout_range = layout_range.data_ptr<int64_t>();
    a.packed_recv_count = recv_count.data_ptr<int>();
    a.num_tokens = x.size(0);
    EPA_CUDA_CHECK(p.H == 7168 and p.K <= 8 ? cudaSuccess : cudaErrorInvalidValue);
    launch_coop(p.fp8 ? (const void*)epa_pull::dispatch<true, 7168> : (const void*)epa_pull::dispatch<false, 7168>,
                p.num_blocks, a);
}

void pull_combine(int64_t id, torch::Tensor topk_idx, torch::Tensor topk_weights, torch::Tensor combined_x) {
    auto& p = g_pull.at(id);
    auto a = pull_args(p);
    a.topk_idx = topk_idx.data_ptr<int64_t>();
    a.topk_weights = topk_weights.data_ptr<float>();
    a.combined_x = reinterpret_cast<nv_bfloat16*>(combined_x.data_ptr());
    a.num_tokens = combined_x.size(0);
    launch_coop((const void*)epa_pull::combine<7168>, p.num_blocks, a);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &create);
    m.def("handle", &handle);
    m.def("open", &open_peers);
    m.def("destroy", &destroy);
    m.def("stats", &stats);
    m.def("arena_stats", &arena_stats);
    m.def("trace", &trace);
    m.def("pull_create", &pull_create);
    m.def("pull_handle", &pull_handle);
    m.def("pull_open", &pull_open);
    m.def("pull_y", &pull_y);
    m.def("pull_destroy", &pull_destroy);
    m.def("pull_dispatch", &pull_dispatch);
    m.def("pull_combine", &pull_combine);
    m.def("dispatch", &dispatch);
    m.def("combine", &combine);
}
