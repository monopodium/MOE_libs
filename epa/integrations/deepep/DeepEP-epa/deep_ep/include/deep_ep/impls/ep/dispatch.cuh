#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/ep/epa.cuh>
#include <deep_ep/layout/ep/epa_out.cuh>
#include <deep_ep/layout/ep/epa_seg.cuh>
#include <deep_ep/layout/ep/token.cuh>
#include <deep_ep/layout/ep/workspace.cuh>


namespace deep_ep::ep {

template <bool kIsScaleupNVLink,
          bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kEPACap, int kEPAPool, int64_t kEPATables,   // EPArena: tier-1 slots per source (-1: upstream layout), tier-2 pool rows
          int kEPAOutP, int kEPAOutPages,                  // paged expanded output (epa_out.cuh); 0: off
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumDispatchThreads = kNumDispatchWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumDispatchThreads,
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int rank_idx,
    int* epa_out_meta, char* epa_out_arena
) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();

    // EPA arena state of SM 0 (cap < -1: arena mode, see layout/ep/epa.cuh)
    constexpr bool kEPAArena = kEPACap < -1;
    const auto epa_layout = layout::EPALayout(
        layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true).get_num_bytes<false, int64_t>(),
        kNumRanks, kNumMaxTokensPerRank, kEPACap == -1 ? 0 : kEPACap, kEPAPool, kEPATables, kNumHiddenBytes);

    // Workspaces
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    const auto host_workspace_layout = layout::EPWorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % kNumTMAAlignmentBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // Gin handle
    // We treat each warp as a "channel"
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
        sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Barrier without TMA store flush, without prologue grid sync
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // EPA arena, step 1 (SM 0): publish the pages this rank already holds on each
        // destination (kept across calls, see layout/ep/epa.cuh), so dispatch warps can send
        // into them before the count is known. No remote atomics: `have` and the page ids
        // were written by the destinations before this call's start barrier
        if constexpr (kEPAArena) {
            if (sm_idx == 0) {
                for (int d = thread_idx; d < kNumRanks; d += kNumNotifyThreads) {
                    const int have = cuda::atomic_ref<int, cuda::thread_scope_system>(epa_layout.have(buffer)[d])
                        .load(cuda::memory_order_acquire);
                    cuda::atomic_ref<int, cuda::thread_scope_device>(epa_layout.avail(buffer)[d]).store(
                        have, cuda::memory_order_release);
                }
            }
        }

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Do full-grid reduction
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>(true, [=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // Write into shared memory
                        // Write into send buffer if with RDMA
                        const auto encoded =
                            math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));
                        rank_expert_count[i] = encoded;
                        if constexpr (not kIsScaleupNVLink)
                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP notify (GPU reduction) timeout, rank: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               rank_idx, kNumRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // EPA tier 2: reserve each destination's overflow (count - cap) in its pool
            // with one relaxed NVLink fetch-add, tell the destination where it is, and
            // publish it to this rank's dispatch warps (`pub`, base + 2)
            if constexpr (kEPACap >= 0) {
                EP_STATIC_ASSERT(kIsScaleupNVLink, "EPArena needs the direct NVLink path");
                const auto epa = layout::EPALayout(
                    layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true).get_num_bytes<false, int64_t>(),
                    kNumRanks, kNumMaxTokensPerRank, kEPACap, kEPAPool, kEPATables);
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    const int over = static_cast<int>(epa.win.overflow(math::encode_decode_positive(rank_count[i])));
                    int base = -1;
                    if (over > 0 and epa.win.shared > 0)
                        base = static_cast<int>(epa.win.accept(atomicAdd_system(gin.get_sym_ptr<team_t>(epa.head(buffer), i), over), over));
                    if (over > 0)
                        *gin.get_sym_ptr<team_t>(epa.tbl(buffer) + rank_idx, i) = base;
                    epa.last(buffer)[i] = base;
                    *reinterpret_cast<volatile int*>(epa.pub(buffer) + i) = base + 2;
                }
            } else if constexpr (kEPAArena) {
                // EPA arena, step 2: final pages per destination = min(need, held); rows beyond
                // stay in Home. Post what was used (`taken`) and needed (`demand`): the destination
                // grows or trims this rank's pages after its copy epilogue
                const int P = epa_layout.rows_per_page;
                for (int d = thread_idx; d < kNumRanks; d += kNumNotifyThreads) {
                    const int need = (math::encode_decode_positive(rank_count[d]) + P - 1) / P;
                    const int have = epa_layout.have(buffer)[d];
                    const int final_pages = need < have ? need : have;
                    *gin.get_sym_ptr<team_t>(epa_layout.taken(buffer) + rank_idx, d) = final_pages;
                    *gin.get_sym_ptr<team_t>(epa_layout.demand(buffer) + rank_idx, d) = need;
                    epa_layout.last_need(buffer)[d] = need;
                    cuda::atomic_ref<int, cuda::thread_scope_device>(epa_layout.avail(buffer)[d]).store(
                        final_pages | layout::EPALayout::kFinal, cuda::memory_order_release);
                }
            }

            // TODO: for further optimization, we can fuse rank and expert counters
            // Issue scaleup rank count writes to peers
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                // Rank counters
                const auto dst_rank_counter =
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                gin.put_value<team_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            __syncwarp();

            // Issue scaleup expert count writes to peers
            if constexpr (kIsScaleupNVLink) {
                // NVLink per-element copy
                // We don't use TMA as the dtype of shared memory and global is different
                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {
                    const auto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);
                    gin.put_value<team_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), i / kNumExpertsPerRank);
                }
            } else {
                // RDMA bulk copy
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    const auto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>() + kNumExpertsPerRank * i;
                    const auto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>() + kNumExpertsPerRank * rank_idx;
                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
                }
            }

            // This is necessary, as the waited results will rewrite the shared memory
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Wait for rank and expert count
            const auto start_clock = clock64();
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    // NOTES: the global memory type has 64 bits
                    const auto count = static_cast<int>(
                        ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));
                    const auto decoded = math::encode_decode_positive(count);
                    if (math::is_decoded_positive_ready(decoded)) {
                        workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;
                        rank_expert_count[i] = decoded;
                        return true;
                    }

                    if (is_last_check)
                        printf("DeepEP notify timeout, rank: %d, thread: %d, count: %d\n", rank_idx, i, decoded);
                    return false;
                }, start_clock);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Reduce expert count and add stats
            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {
                int sum = 0;
                #pragma unroll
                for (int j = 0; j < kNumRanks; ++ j)
                    sum += expert_count[j * kNumExpertsPerRank + i];

                // Write unaligned count before aligning
                if (num_unaligned_recv_tokens_per_expert != nullptr)
                    num_unaligned_recv_tokens_per_expert[i] = sum;

                expert_count[i] = math::align(sum, kExpertAlignment);

                // Update statistics counters
                if (cumulative_local_expert_recv_stats != nullptr)
                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // EPArena segment (layout/ep/epa_seg.cuh): publish this GPU's end, then claim the rows
            // beyond C from the lenders' tails (thread 0; lenders never wait, so no cycle)
            if constexpr (kEPAOutP == 0) {
                if (epa_out_meta != nullptr and thread_idx == 0) {
                    using sys_int = cuda::atomic_ref<int, cuda::thread_scope_system>;
                    int* hd = epa_out_meta;
                    const auto seg = layout::EPASeg{hd};
                    int total = 0;
                    for (int i = 0; i < kNumExpertsPerRank; ++ i) total += expert_count[i];
                    const int C = hd[0], e = hd[1] + 1, own_end = min(total, C);
                    hd[5] = total, hd[2] = own_end;
                    sys_int(hd[3]).store(own_end, cuda::memory_order_relaxed);
                    __threadfence_system();
                    sys_int(hd[1]).store(e, cuda::memory_order_release);
                    const int need = max(0, total - C);
                    int got = 0, n = 0;
                    if (const int spill = min(need, hd[8]); spill > 0) {   // local spill scratch first: no NVLink
                        int* ch = hd + 16;
                        ch[0] = kNumRanks, ch[1] = C, ch[2] = spill, ch[3] = 0;
                        n = 1, got = spill;
                    }
                    for (int k = 1; k < kNumRanks and got < need and n < layout::EPASeg::kMaxChunks; ++ k) {
                        const int d = (rank_idx + k) % kNumRanks;
                        int* ph = reinterpret_cast<int*>(reinterpret_cast<char*>(seg.bases()[d]) - layout::EPASeg::kHeaderBytes);
                        sys_int pe(ph[1]), pc(ph[3]);
                        while (pe.load(cuda::memory_order_acquire) < e) __nanosleep(64);
                        const int cd = ptx::ld_volatile<int>(ph);
                        int c = pc.load(cuda::memory_order_relaxed), q = 0;
                        while ((q = min(need - got, cd - c)) > 0 and not pc.compare_exchange_weak(c, c + q, cuda::memory_order_relaxed)) {}
                        if (q > 0) {
                            int* ch = hd + 16 + 4 * n;
                            ch[0] = d, ch[1] = C + got, ch[2] = q, ch[3] = c;
                            ++ n, got += q;
                        }
                    }
                    hd[4] = n, hd[6] = need - got, hd[7] += max(0, got - min(need, hd[8]));
                    __threadfence_system();
                }
            }

            // Paged expanded output (layout/ep/epa_out.cuh): sticky pages per local expert;
            // only the difference to the expert's held pages is taken or given back
            if constexpr (kEPAOutP != 0) {
                // kEPAOutP > 0: expanded layout, pages per local expert (aligned expert counts);
                // kEPAOutP < 0: normal layout, pages per source rank (received token counts).
                // Each tile holds a recv_x page and a y page. Shared pool (layout/ep/epa.cuh): recv_x
                // pages from zone A (else B), y pages from zone B; otherwise one output ring, y page = x page
                namespace eo = layout::epa_out;
                constexpr bool kNormal = kEPAOutP < 0;
                constexpr bool kShared = kEPACap < -1 and kEPAPool < 0;
                constexpr int kP = kNormal ? -kEPAOutP : kEPAOutP;
                constexpr int N = kEPAOutPages, L = kNormal ? kNumRanks : kNumExpertsPerRank;
                const int* key_count = kNormal ? rank_count : expert_count;
                const auto out = layout::EPAOutLayout{epa_out_meta, L, N, kP};
                const auto ra = kShared ? epa_layout.arena(buffer) : out.arena(epa_out_arena, kNumHiddenBytes);
                const auto rb = kShared ? epa_layout.arena_b(buffer) : ra;
                const int na = kShared ? epa_layout.num_pages : 0;
                const auto give1 = [&](const int page) {   // one page back to its ring
                    const bool in_b = kShared and page >= na;
                    const auto r = in_b ? rb : ra;
                    const auto t0 = eo::dev_ref<unsigned long long>(r.meta()->tail).fetch_add(1, cuda::memory_order_relaxed);
                    eo::give_slot(r, t0, 0, in_b ? page - na : page);
                    eo::dev_ref<int>(r.meta()->free_pages).fetch_add(1, cuda::memory_order_release);
                };
                const auto give = [&](const int e, const int from, const int to) {   // tiles [from, to) of e
                    for (int q = from; q < to; ++ q) {
                        const int px = out.etable()[e * N + q];
                        out.tile_expert()[px] = -1;
                        give1(px);
                        if constexpr (kShared) give1(out.etable_y()[e * N + q]);
                    }
                    atomicAdd(out.meta + 6, to - from);
                };
                // Peer spill: receive pages first. A short receive page costs one NVLink read (the row stays
                // in its sender's Home), a borrowed page four transfers, so recv_x / y takes and lending leave
                // free the pages the receive manager grows this call (zone A first, then B)
                int floor_a = 0, floor_b = 0;
                if constexpr (kShared) {
                    if (out.meta[13] > 0) {
                        int grow = 0;
                        for (int src = 0; src < kNumRanks; ++ src)
                            grow += max(0, (rank_count[src] + epa_layout.rows_per_page - 1) / epa_layout.rows_per_page -
                                           epa_layout.held(buffer)[src]);
                        const int free_a = eo::dev_ref<int>(ra.meta()->free_pages).load(cuda::memory_order_relaxed);
                        floor_a = min(grow, free_a);
                        floor_b = grow - floor_a + epa_layout.num_pages_b / 16;
                    }
                }
                const auto take_from = [&](const epa::Arena& r, const int base, int* dst, const int n) {
                    const long long t = eo::try_take(r, n, base == 0 ? floor_a : floor_b);
                    if (t < 0) return false;
                    for (int i = 0; i < n; ++ i)
                        dst[i] = base + eo::page_at(r, static_cast<unsigned long long>(t), i);
                    return true;
                };
                const auto take = [&](const int e, const int h, const int n) {   // n tiles appended at h; false if short
                    int* dx = out.etable() + e * N + h;
                    int* dy = out.etable_y() + e * N + h;
                    if constexpr (kShared) {
                        if (not take_from(ra, 0, dx, n) and not take_from(rb, na, dx, n)) return false;
                        if (not take_from(rb, na, dy, n)) {
                            for (int i = 0; i < n; ++ i) give1(dx[i]);
                            return false;
                        }
                    } else {
                        if (not take_from(ra, 0, dx, n)) return false;
                        for (int i = 0; i < n; ++ i) dy[i] = dx[i];
                    }
                    atomicAdd(out.meta + 7, n);
                    return true;
                };
                // Peer spill (shared pool, window view set up by the host): take back the pages lent last
                // call (every borrower finished that call's combine before sending this call's counts)
                const bool spill = kShared and out.meta[13] > 0;
                int* lend = kShared ? epa_layout.lend_hdr(buffer) : nullptr;
                if constexpr (kShared) {
                    if (spill) {
                        // Own pages of last call's partly borrowed tiles
                        const int n_tmp = *out.tmp_n();
                        for (int i = thread_idx; i < n_tmp; i += kNumNotifyThreads) give1(out.tmp()[i]);
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        if (thread_idx == 0) *out.tmp_n() = 0;
                        const int n_prev = ptx::ld_volatile<int>(lend + 2);
                        if (n_prev > 0) {
                            auto* tk = reinterpret_cast<unsigned long long*>(lend + 4);
                            if (thread_idx == 0)
                                *tk = eo::dev_ref<unsigned long long>(rb.meta()->tail).fetch_add(n_prev, cuda::memory_order_relaxed);
                            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                            const auto tb = *reinterpret_cast<volatile unsigned long long*>(tk);
                            for (int i = thread_idx; i < n_prev; i += kNumNotifyThreads)
                                eo::give_slot(rb, tb, i, epa_layout.lend_pages(buffer)[i] - na);
                            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                            if (thread_idx == 0) {
                                eo::dev_ref<int>(rb.meta()->free_pages).fetch_add(n_prev, cuda::memory_order_release);
                                lend[2] = 0;
                            }
                            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        }
                    }
                }
                // Pass 1: trim the generous, grow the short (each thread its keys).
                // Tight pool (under 1/8 free at entry): no slack, trim to the need
                bool short_any = false;
                // Peer spill: no slack either, so every page beyond a key's need is free to lend
                const bool tight = kShared ?
                    spill or eo::dev_ref<int>(rb.meta()->free_pages).load(cuda::memory_order_relaxed) < epa_layout.num_pages_b / 8 :
                    eo::dev_ref<int>(ra.meta()->free_pages).load(cuda::memory_order_relaxed) < N / 8;
                for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                    const int need = (key_count[e] + kP - 1) / kP;
                    const int slack = tight ? 0 : 1 + need / 8, keep = min(need + slack, N);
                    int h = out.held_e()[e];
                    if (h > keep + slack)
                        give(e, keep, h), h = keep;
                    if (h < need) {
                        if (take(e, h, keep - h)) h = keep;
                        else if (take(e, h, need - h)) h = need;
                        else short_any = true;
                    }
                    out.held_e()[e] = h;
                }
                if (short_any) *out.flag() = 1;
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                // Pass 2 (rare, pool short): trim every key to its need, then retry the short
                if (thread_idx == 0) atomicAdd(out.meta + 5, 1);
                if (*out.flag()) {
                    if (thread_idx == 0) atomicAdd(out.meta + 10, 1);
                    for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                        const int need = (key_count[e] + kP - 1) / kP;
                        const int h = out.held_e()[e];
                        if (h > need) give(e, need, h), out.held_e()[e] = need;
                    }
                    ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                    for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                        const int need = (key_count[e] + kP - 1) / kP;
                        const int h = out.held_e()[e];
                        if (h < need) {
                            if (take(e, h, need - h)) out.held_e()[e] = need;
                            else if (not spill) atomicAdd(out.failed(), 1);
                        }
                    }
                    ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                    if (thread_idx == 0) *out.flag() = 0;
                }
                // Peer spill: lend the free zone-B pages (but a reserve for receive-page growth) for this
                // call, then borrow peers' lent pages for the keys still short. Borrowed tiles sit after
                // the key's own tiles, encoded -(2 + global id); cnt[e] = borrowed tiles (until marks)
                if constexpr (kShared) {
                    if (spill) {
                        int* lp = epa_layout.lend_pages(buffer);
                        if (thread_idx == 0) {
                            const int nf = eo::dev_ref<int>(rb.meta()->free_pages).load(cuda::memory_order_relaxed);
                            const int n = max(0, nf - floor_b);
                            const long long t = n > 0 ? eo::try_take(rb, n) : -1;
                            lend[2] = t >= 0 ? n : 0;
                            *reinterpret_cast<unsigned long long*>(lend + 4) = t >= 0 ? static_cast<unsigned long long>(t) : 0;
                            lend[1] = 0;
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        {
                            const int n = ptx::ld_volatile<int>(lend + 2);
                            const auto tb = *reinterpret_cast<volatile unsigned long long*>(lend + 4);
                            for (int i = thread_idx; i < n; i += kNumNotifyThreads)
                                lp[i] = na + eo::page_at(rb, tb, i);
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        if (thread_idx == 0) {
                            __threadfence_system();
                            const int ep = lend[0] + 1;
                            cuda::atomic_ref<int, cuda::thread_scope_system>(lend[0]).store(ep, cuda::memory_order_release);
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        const int my_epoch = ptx::ld_volatile<int>(lend);
                        const int pps = out.meta[13];
                        // Short keys, tile by tile: the recv_x page from zone A and the y page from zone B where
                        // free (a borrowed recv_x page costs less than a borrowed y page: combine reads y row by row);
                        // a missing page is -1, and cnt[e] = missing pages (temporarily). Own pages of these tiles
                        // go to tmp (given back next call)
                        // y pages first (zone B only), then recv_x pages (zone A, else what B has left)
                        for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                            const int need = (key_count[e] + kP - 1) / kP, h = out.held_e()[e];
                            int miss = 0;
                            for (int q = h; q < need; ++ q) {
                                int py;
                                if (take_from(rb, na, &py, 1)) out.tmp()[atomicAdd(out.tmp_n(), 1)] = py; else py = -1, ++ miss;
                                out.etable_y()[e * N + q] = py;
                            }
                            out.cnt()[e] = miss;
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                            const int need = (key_count[e] + kP - 1) / kP, h = out.held_e()[e];
                            int miss = 0;
                            for (int q = h; q < need; ++ q) {
                                int px;
                                if (take_from(ra, 0, &px, 1) or take_from(rb, na, &px, 1))
                                    out.tmp()[atomicAdd(out.tmp_n(), 1)] = px;
                                else px = -1, ++ miss;
                                out.etable()[e * N + q] = px;
                            }
                            out.cnt()[e] += miss;
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        // Thread 0 claims the missing pages from the lenders in turn (one CAS each), recording
                        // (lender, first, pages) in lend[8 + 3j]; all threads read the claimed ids in parallel into
                        // tiles (free until the dense lists)
                        EP_STATIC_ASSERT(kNumRanks <= 18, "peer spill: lend header holds 18 claims");
                        if (thread_idx == 0) {
                            int want = 0, got = 0, nc = 0;
                            for (int e = 0; e < L; ++ e) want += out.cnt()[e];
                            want = min(want, N);
                            for (int k = 1; k < kNumRanks and got < want; ++ k) {
                                const int d = (rank_idx + k) % kNumRanks;
                                int* rh = gin.get_sym_ptr<team_t>(lend, d);
                                cuda::atomic_ref<int, cuda::thread_scope_system> rep(rh[0]), rclaim(rh[1]);
                                while (rep.load(cuda::memory_order_acquire) < my_epoch) __nanosleep(64);
                                const int n_d = ptx::ld_volatile<int>(rh + 2);
                                int c = rclaim.load(cuda::memory_order_relaxed), q = 0;
                                while ((q = min(want - got, n_d - c)) > 0 and
                                       not rclaim.compare_exchange_weak(c, c + q, cuda::memory_order_relaxed)) {}
                                if (q > 0) {
                                    lend[8 + 3 * nc] = d, lend[9 + 3 * nc] = c, lend[10 + 3 * nc] = q;
                                    ++ nc, got += q;
                                }
                            }
                            lend[6] = got, lend[7] = nc;
                            if (got > 0) atomicAdd(out.meta + 11, got);
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        const int got = ptx::ld_volatile<int>(lend + 6);
                        for (int k = thread_idx; k < got; k += kNumNotifyThreads) {
                            int j = 0, first = 0;
                            while (first + ptx::ld_volatile<int>(lend + 10 + 3 * j) <= k) first += ptx::ld_volatile<int>(lend + 10 + 3 * j), ++ j;
                            const int d = ptx::ld_volatile<int>(lend + 8 + 3 * j), c = ptx::ld_volatile<int>(lend + 9 + 3 * j);
                            out.tiles()[1 + k] = -2 - (d * pps + ptx::ld_volatile<int>(gin.get_sym_ptr<team_t>(lp, d) + c + k - first));
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                        // Each key fills its missing pages in tile order from its share; complete tiles form a prefix
                        // (cnt[e] = tiles past the held ones, for the marks)
                        for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                            int off = 0;
                            for (int k = 0; k < e; ++ k) off += out.cnt()[k];
                            const int need = (key_count[e] + kP - 1) / kP, h = out.held_e()[e];
                            int share = max(0, min(out.cnt()[e], got - off)), k = 0, done = h;
                            for (int q = h; q < need; ++ q) {
                                int& px = out.etable()[e * N + q];
                                int& py = out.etable_y()[e * N + q];
                                if (px == -1 and k < share) px = out.tiles()[1 + off + k ++];
                                if (py == -1 and k < share) py = out.tiles()[1 + off + k ++];
                                if (px != -1 and py != -1 and done == q) ++ done;
                            }
                            if (done < need) atomicAdd(out.failed(), 1);
                            __syncwarp(__activemask());
                            out.cnt()[e] = done - h;
                        }
                        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                    }
                }
                // Marks: own tiles [0, lvalid) of e are e's, [lvalid, held) are free tiles; avail adds the
                // borrowed tiles
                for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                    const int need = (key_count[e] + kP - 1) / kP;
                    const int h = out.held_e()[e], v = out.lvalid()[e];
                    const int nv = min(need, h);
                    for (int q = v; q < nv; ++ q) out.tile_expert()[out.etable()[e * N + q]] = e;
                    for (int q = nv; q < min(v, h); ++ q) out.tile_expert()[out.etable()[e * N + q]] = -1;
                    out.lvalid()[e] = nv;
                    out.avail()[e] = nv + (kShared and spill ? out.cnt()[e] : 0);
                    out.cnt()[e] = 0;
                }
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
                // Dense tile lists (global page ids): key e's valid tiles at offset sum(avail[< e])
                for (int e = thread_idx; e < L; e += kNumNotifyThreads) {
                    int off = 0;
                    for (int k = 0; k < e; ++ k) off += out.avail()[k];
                    const int nv = out.avail()[e];
                    for (int q = 0; q < nv; ++ q) {
                        out.tiles()[1 + off + q] = out.gid(out.etable()[e * N + q]);
                        out.tiles_y()[1 + off + q] = out.gid(out.etable_y()[e * N + q]);
                    }
                    if (e == L - 1) out.tiles()[0] = out.tiles_y()[0] = off + nv;
                }
                ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
            }

            // Write host workspace
            if constexpr (kDoCPUSync) {
                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {
                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =
                        math::encode_decode_positive(rank_expert_count[i]);
                }
                __syncwarp();
            }

            // Do prefix sum by the warps
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else {
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

        // EPA arena, cached mode (no notify warps): the counts are last call's
        if constexpr (kEPAArena and kReuseSlotIndices) {
            if (sm_idx == 0 and dispatch_warp_idx == 0) {
                for (int d = lane_idx; d < kNumRanks; d += 32) {
                    const int need = epa_layout.last_need(buffer)[d];
                    const int have = cuda::atomic_ref<int, cuda::thread_scope_system>(epa_layout.have(buffer)[d])
                        .load(cuda::memory_order_acquire);
                    const int final_pages = need < have ? need : have;
                    *gin.get_sym_ptr<team_t>(epa_layout.taken(buffer) + rank_idx, d) = final_pages;
                    *gin.get_sym_ptr<team_t>(epa_layout.demand(buffer) + rank_idx, d) = need;
                    cuda::atomic_ref<int, cuda::thread_scope_device>(epa_layout.avail(buffer)[d]).store(
                        final_pages | layout::EPALayout::kFinal, cuda::memory_order_release);
                }
                __syncwarp();
            }
        }

        // Buffer layouts
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
        const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(dispatch_warp_idx).get_token_buffer(0);
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
        const auto& epa = epa_layout;
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());
        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);

        // Init TMA
        ptx::arrival_phase phase = 0;
        const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
        if (ptx::elect_one_sync())
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        __syncwarp();

        // Iterate all tokens
        const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
        const auto token_stride = kNumDispatchWarps * kNumSMs;
        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
            const auto token_i64_idx = static_cast<int64_t>(token_idx);

            // Wait TMA store arrivals
            ptx::tma_store_wait();
            __syncwarp();

            // Issue data TMA
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF TMA or cp.async
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }

            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto dst_expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            // Please ensure no TMA buffer shared memory writes after this part
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            if constexpr (kReuseSlotIndices) {
                if (lane_idx < kNumTopk)
                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
                stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
                    (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
            } else {
                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
                if (lane_idx < kNumTopk) {
                    const auto value = stored_dst_slot_idx >= 0 ?
                        rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;
                }
            }
            __syncwarp();

            // Wait TMA load arrival
            // NOTES: this arrive must be after the `ptx::cp_async_mbarrier_arrive`
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            // TMA store to send buffer
            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();
            if constexpr (not kIsScaleupNVLink) {
                if (ptx::elect_one_sync())
                    ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                ptx::tma_store_commit();
                __syncwarp();
            }

            // Issue TMA NVLink stores
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");
            void* dst_ptr = nullptr;
            if constexpr (kEPACap == -1) {
                dst_ptr = stored_dst_slot_idx >= 0 ?
                    gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                    nullptr;
            } else {
                // EPA window: keyed row, else shared row, else the token stays in Home.
                // EPA arena: page slot / P of the destination's arena once published.
                bool spill = false;
                if constexpr (kEPAArena) {
                    if (stored_dst_slot_idx >= 0) {
                        const int p = stored_dst_slot_idx / epa.rows_per_page;
                        cuda::atomic_ref<int, cuda::thread_scope_device> av(epa.avail(buffer)[stored_dst_rank_idx]);
                        int a;
                        while (((a = av.load(cuda::memory_order_acquire)) & ~layout::EPALayout::kFinal) <= p and
                               not (a & layout::EPALayout::kFinal))
                            __nanosleep(32);
                        if ((a & ~layout::EPALayout::kFinal) > p) {
                            const int id = epa.ltable(buffer)[stored_dst_rank_idx * epa.max_pages + p];
                            void* local = epa.arena(buffer).page_ptr(id) +
                                static_cast<int64_t>(stored_dst_slot_idx % epa.rows_per_page) * epa.token_bytes;
                            dst_ptr = gin.get_sym_ptr<team_t>(local, stored_dst_rank_idx);
                        } else {
                            spill = true;
                        }
                    }
                } else if (stored_dst_slot_idx >= 0) {
                    int base = -1;
                    if (stored_dst_slot_idx >= epa.win.keyed) {
                        if constexpr (kReuseSlotIndices) {
                            base = epa.last(buffer)[stored_dst_rank_idx];
                        } else {
                            const volatile int* pub = epa.pub(buffer) + stored_dst_rank_idx;
                            while ((base = *pub) == 0) __nanosleep(32);
                            base -= 2;
                        }
                    }
                    void* local = epa.row(buffer, rank_idx, stored_dst_slot_idx, base);
                    if (local != nullptr)
                        dst_ptr = gin.get_sym_ptr<team_t>(local, stored_dst_rank_idx);
                    else
                        spill = true;
                }
                if (__ballot_sync(0xffffffff, spill)) {
                    // One Home copy per token, one 4-byte descriptor per destination without a window row
                    if (ptx::elect_one_sync())
                        ptx::tma_store_1d(epa.home(buffer, token_idx), tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    if (spill)
                        *gin.get_sym_ptr<team_t>(epa.desc(buffer, rank_idx, stored_dst_slot_idx), stored_dst_rank_idx) = token_idx;
                }
            }
            if (dst_ptr != nullptr)
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            ptx::tma_store_commit();
            __syncwarp();

            // Issue RDMA put
            if constexpr (not kIsScaleupNVLink) {
                // Wait the send buffer store to arrive
                ptx::tma_store_wait<1>();
                __syncwarp();

                // NOTES: we should skip the NVLink accessible ranks
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                __syncwarp();
            }
        }
    }

    // Barrier to ensure data arrival
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Clean atomic counters
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;

    // EPA: every source has reserved in this rank's pool and every local warp has
    // read `pub` by now (final barrier), so both can be reset for the next call
    if constexpr (kEPACap >= 0) {
        const auto epa_bytes = layout::EPALayout(
            layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true).get_num_bytes<false, int64_t>(),
            kNumRanks, kNumMaxTokensPerRank, kEPACap, kEPAPool, kEPATables);
        if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks) {
            epa_bytes.pub(buffer)[thread_idx] = 0;
            if (thread_idx == 0)
                *epa_bytes.head(buffer) = 0;
        }
    }
    // EPA arena: every local send has read `avail` by now (final barrier)
    if constexpr (kEPAArena) {
        if (sm_idx == 0 and thread_idx < kNumRanks)
            epa_layout.avail(buffer)[thread_idx] = 0;
    }
}

}  // namespace deep_ep::ep
