"""Derive the EPArena versions of Mooncake's legacy (low-latency) kernels from the
upstream headers, as a small reviewable patch. Run after updating Mooncake.

Changes (both kernels keep upstream's warp roles, signals and launch shape):

dispatch
  * recv region [L][R][T] messages  ->  tier-1 slots [L][R][cap], a tier-2 pool
    of `pool` rows shared by every (expert, source) pair of the receiver, and
    4-byte descriptors [L][R][T].
      tier 1  slot j < cap is pushed exactly as upstream.
      tier 2  the sender's count warp (which already counts tokens per expert)
              reserves the pair's overflow n - cap in the receiver's pool with
              one NVLink fetch-add, and data warps push slots j >= cap there.
      tier 3  if the pool is full, the message stays in the sender's home row
              (the FP8/BF16 staging row upstream already writes) and only
              (k << 24 | token) is pushed; the receiver pulls the row.
    The pool head lives after the receive signals and is cleared with them by
    the next call, like upstream's `next_clean_buffer`.
  * the top-k position k travels with the message (header int 1 / descriptor),
    and packed_recv_src_info stores (k << 24 | token).

combine
  * recv region [E][T] slots keyed by expert  ->  [K][T] keyed by top-k
    position, which is exact (each token has K results), so no capacity check
    or fallback is needed.
"""
from pathlib import Path

UP = Path('/home/Shuang/shuangma/mkernel_project/MOE_libs/Mooncake/mooncake-ep/include')
HERE = Path(__file__).parent


def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f'expected {count} match(es), got {n}:\n{old}'
    return s.replace(old, new)


def namespace(s):
    s = sub(s, 'namespace mooncake {\n', '#include <epa/window.cuh>\n\nnamespace mooncake_epa {\n\nusing namespace mooncake;\n')
    return sub(s, '}  // namespace mooncake\n', '}  // namespace mooncake_epa\n')


# ----------------------------------------------------------------------- dispatch
d = (UP / 'mooncake_ep_dispatch.cuh').read_text()
d = namespace(d)
d = sub(d, '    int active_qps_per_rank;\n};\n\nstatic_assert(std::is_trivially_copyable<DispatchKernelArgs>',
        '    int active_qps_per_rank;\n'
        '    // EPArena: tier-1 slots per (local expert, source rank) and descriptor table\n'
        '    epa::Window epa_win;   // keys = (local expert, source); keyed = tier-1 rows, shared = pool\n'
        '    int* epa_recv_desc;\n'
        '    int* epa_pool_head;\n'
        '    int* epa_pool_tbl;\n'
        '    int* epa_pool_base_local;\n'
        '    int* epa_stats;   // [2] messages placed in tier 2 / tier 3 (receiver side)\n'
        '};\n\nstatic_assert(std::is_trivially_copyable<DispatchKernelArgs>')
d = sub(d, '    int active_qps_per_rank = args.active_qps_per_rank;\n',
        '    int active_qps_per_rank = args.active_qps_per_rank;\n'
        '    const epa::Window epa_win = args.epa_win;\n'
        '    int* epa_recv_desc = args.epa_recv_desc;\n'
        '    int* epa_pool_head = args.epa_pool_head;\n'
        '    int* epa_pool_tbl = args.epa_pool_tbl;\n'
        '    int* epa_pool_base_local = args.epa_pool_base_local;\n'
        '    int* epa_stats = args.epa_stats;\n')
# sender: tier choice
d = sub(d, '''                    const auto dst_ptr = reinterpret_cast<void*>(
                        reinterpret_cast<uint64_t>(rdma_recv_data_buffer) +
                        dst_expert_local_idx * num_ranks *
                            num_max_dispatch_tokens_per_rank *
                            num_bytes_per_msg +
                        rank * num_max_dispatch_tokens_per_rank *
                            num_bytes_per_msg +
                        slot_idx * num_bytes_per_msg);
''', '''                    // EPA window: keyed row, else shared row, else -1 (stays in Home)
                    const int epa_key = dst_expert_local_idx * num_ranks + rank;
                    int64_t epa_base = -1;
                    if (slot_idx >= epa_win.keyed) {
                        int pool_base = 0;
                        if (lane_id == 0) {
                            const volatile int* pb = epa_pool_base_local + dst_expert_idx;
                            while ((pool_base = *pb) == 0) __nanosleep(32);
                        }
                        epa_base = __shfl_sync(0xffffffff, pool_base, 0) - 2;
                    }
                    const int64_t epa_row = epa_win.row(epa_key, slot_idx, epa_base);
                    if (epa_row < 0) {
                        // EPA: the message stays in the Home row, push its descriptor
                        if (lane_id == 0) {
                            int* desc = epa_recv_desc +
                                static_cast<int64_t>(epa_key) * num_max_dispatch_tokens_per_rank + slot_idx;
                            *reinterpret_cast<int*>(mc_route_put(comm_ctx, dst_rank, desc)) =
                                (warp_id << 24) | token_idx;
                        }
                    } else {
                    const auto dst_ptr = reinterpret_cast<void*>(
                        reinterpret_cast<uint64_t>(rdma_recv_data_buffer) +
                        epa_row * num_bytes_per_msg);
''')
d = sub(d, '''                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg,
                                           dst_int4_ptr, src_int4_ptr, mc_ld_nc,
                                           mc_st_na);
''', '''                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg,
                                           dst_int4_ptr, src_int4_ptr, mc_ld_nc,
                                           mc_st_na);
                        // EPA: the top-k position travels in header int 1
                        if (lane_id == 0)
                            reinterpret_cast<int*>(write_dst)[1] = warp_id;
''')
d = sub(d, "    if (is_count_worker) {\n        EP_DEVICE_ASSERT(num_sms > 1);\n        if (sm_id == 0) {\n// The first SM is also responsible for cleaning the next buffer\n#pragma unroll\n            for (int i = lane_id; i < qp_num_ranks * num_local_experts; i += 32)\n                next_clean_buffer[i] = 0;\n\n            // Notify before executing `int_p`\n            __syncwarp();\n#ifdef MOONCAKE_EP_USE_MUSA\n            // Publish the cleared buffer once before any local ready tag.\n            mc_fence();\n            __syncwarp();\n#endif\n#pragma unroll\n            for (int i = lane_id; i < num_experts; i += 32)\n#ifdef MOONCAKE_EP_USE_MUSA\n                mc_atomic_add_relaxed(atomic_finish_counter_per_expert + i,\n                                      FINISHED_SUM_TAG);\n#else\n                mc_atomic_add_release(atomic_finish_counter_per_expert + i,\n                                      FINISHED_SUM_TAG);\n#endif\n        }\n\n        // This SM should be responsible for some destination experts, read\n        // `topk_idx` for them\n        int expert_count[kNumWarpGroups] = {0};\n        const auto expert_begin_idx = sm_id * kNumWarpGroups;\n        const auto expert_end_idx =\n            min(expert_begin_idx + kNumWarpGroups, num_experts);\n\n// Per lane count\n#pragma unroll 8\n        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {\n            auto idx = static_cast<int>(__ldg(topk_idx + i));\n            if (idx >= expert_begin_idx and idx < expert_end_idx)\n                expert_count[idx - expert_begin_idx]++;\n        }\n\n// Warp reduce\n#pragma unroll\n        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {\n            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);\n            if (lane_id == 0) {\n                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;\n#ifdef MOONCAKE_EP_USE_MUSA\n                // Local completion accounting; payload publication is fenced\n                // by the expert's signal sender after the count reaches its\n                // tag.\n                mc_atomic_add_relaxed(atomic_finish_counter_per_expert + i,\n                                      FINISHED_SUM_TAG - sum);\n#else\n                mc_atomic_add_release(atomic_finish_counter_per_expert + i,\n                                      FINISHED_SUM_TAG - sum);\n#endif\n            }\n        }\n", "    if (is_count_worker) {\n        EP_DEVICE_ASSERT(num_sms > 1);\n        // EPA: count first. The sys-scope release operations below wait for this\n        // SM's outstanding NVLink stores, so anything placed after them (like the\n        // tier-2 reservation that data warps wait on) would be serialized behind\n        // the whole tier-1 send.\n        int expert_count[kNumWarpGroups] = {0};\n        const auto expert_begin_idx = sm_id * kNumWarpGroups;\n        const auto expert_end_idx =\n            min(expert_begin_idx + kNumWarpGroups, num_experts);\n\n// Per lane count\n#pragma unroll 8\n        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {\n            auto idx = static_cast<int>(__ldg(topk_idx + i));\n            if (idx >= expert_begin_idx and idx < expert_end_idx)\n                expert_count[idx - expert_begin_idx]++;\n        }\n\n// Warp reduce: lane j keeps the count of expert begin + j\n        int lane_sum = 0;\n#pragma unroll\n        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {\n            const auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);\n            if (lane_id == i - expert_begin_idx) lane_sum = sum;\n        }\n\n        // EPA tier 2: reserve this rank's overflow (count - cap) in the receiver's\n        // pool with one relaxed NVLink fetch-add, tell the receiver where it is,\n        // and publish it to the local data warps (0 = not ready, base + 2).\n        // Relaxed is enough: the base orders no data.\n        const bool owns_expert = lane_id < expert_end_idx - expert_begin_idx;\n        if (owns_expert) {\n            const int i = expert_begin_idx + lane_id;\n            const int dst_rank_ = i / num_local_experts, e_local_ = i % num_local_experts;\n            const int over = static_cast<int>(epa_win.overflow(lane_sum));\n            int base = -1;\n            if (over > 0 and epa_win.shared > 0 and active_ranks[dst_rank_])\n                base = static_cast<int>(epa_win.accept(\n                    atomicAdd_system(reinterpret_cast<int*>(mc_route_put(comm_ctx, dst_rank_, epa_pool_head)), over), over));\n            if (over > 0)\n                *reinterpret_cast<int*>(mc_route_put(comm_ctx, dst_rank_,\n                    epa_pool_tbl + e_local_ * num_ranks + rank)) = base;\n            *reinterpret_cast<volatile int*>(epa_pool_base_local + i) = base + 2;\n        }\n        __syncwarp();\n\n        if (sm_id == 0) {\n// The first SM is also responsible for cleaning the next buffer\n#pragma unroll\n            for (int i = lane_id; i < qp_num_ranks * num_local_experts + 1; i += 32)  // EPA: + pool head\n                next_clean_buffer[i] = 0;\n\n            // Notify before executing `int_p`\n            __syncwarp();\n#pragma unroll\n            for (int i = lane_id; i < num_experts; i += 32)\n                mc_atomic_add_release(atomic_finish_counter_per_expert + i,\n                                      FINISHED_SUM_TAG);\n        }\n\n        if (owns_expert) {\n            shared_num_tokens_sent_per_expert[lane_id] = lane_sum;\n            mc_atomic_add_release(atomic_finish_counter_per_expert + expert_begin_idx + lane_id,\n                                  FINISHED_SUM_TAG - lane_sum);\n        }\n")
d = sub(d, '''        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
''', '''        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
        epa_pool_base_local[responsible_expert_idx] = 0;
''')
d = sub(d, '''#endif
                    }
                }

                // Increase counter after finishing''', '''#endif
                    }
                    }  // EPA tier 1
                }

                // Increase counter after finishing''')
# receiver: resolve tier-1 slot or pull tier-3 row
d = sub(d, '''        const auto rdma_recv_x_uint8 =
            reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
                num_bytes_per_msg +
            src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
''', '''        const int epa_key = local_expert_idx * num_ranks + src_rank;
        const auto rdma_recv_x_uint8 =
            reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) +
            epa_win.row(epa_key, 0) * num_bytes_per_msg;
        const int* epa_desc = epa_recv_desc +
            static_cast<int64_t>(epa_key) * num_max_dispatch_tokens_per_rank;
        const auto epa_peer_home = reinterpret_cast<const uint8_t*>(
            mc_route_put(comm_ctx, src_rank, rdma_send_data_buffer));
''')
d = sub(d, '''        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];
''', '''        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];
        // EPA tier 2: where this pair's overflow landed in the pool (-1: tier 3)
        const int epa_over = static_cast<int>(epa_win.overflow(num_recv_tokens));
        const int epa_pool_base = epa_over > 0 ?
            *reinterpret_cast<const volatile int*>(epa_pool_tbl + epa_key) : -1;
        if (epa_stats != nullptr and sub_warp_id == 0 and lane_id == 0 and epa_over > 0)
            atomicAdd(epa_stats + (epa_pool_base >= 0 ? 0 : 1), epa_over);
''')
d = sub(d, '''            const auto src_src_idx = reinterpret_cast<int*>(
                rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] =
                    mc_ld_nc_s32(src_src_idx);
            __syncwarp();
''', '''            int* src_src_idx;
            const int64_t epa_row = epa_win.row(epa_key, i, epa_pool_base);
            if (epa_row >= 0) {
                src_src_idx = reinterpret_cast<int*>(
                    reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) + epa_row * num_bytes_per_msg);
                if (lane_id == 0)
                    recv_src_info[recv_token_begin_idx + i] =
                        (mc_ld_nc_s32(src_src_idx + 1) << 24) | mc_ld_nc_s32(src_src_idx);
            } else {
                // EPA: read the row in place from the sender's Home over NVLink
                const int desc = *reinterpret_cast<const volatile int*>(epa_desc + i);
                src_src_idx = reinterpret_cast<int*>(const_cast<uint8_t*>(
                    epa_peer_home + static_cast<int64_t>(desc & 0xffffff) * num_bytes_per_msg));
                if (lane_id == 0)
                    recv_src_info[recv_token_begin_idx + i] = desc;
            }
            __syncwarp();
''')
(HERE / 'epa_dispatch.cuh').write_text(d)

# ------------------------------------------------------------------------ combine
c = (UP / 'mooncake_ep_combine.cuh').read_text()
c = namespace(c)
c = sub(c, '''        for (int i = lane_id; i < qp_num_ranks * num_local_experts; i += 32)
''', '''        for (int i = lane_id; i < qp_num_ranks * num_local_experts + 1; i += 32)  // EPA: + pool head
''')
c = sub(c, '''            auto src_idx = __ldg(local_src_info + token_idx);
''', '''            auto src_idx = __ldg(local_src_info + token_idx);
            // EPA: results are keyed by top-k position, not by expert
            const int epa_k = src_idx >> 24;
            src_idx &= 0xffffff;
''')
c = sub(c, '''                (global_expert_idx * num_max_dispatch_tokens_per_rank +
                 src_idx) *
                    num_bytes_per_slot);
''', '''                (static_cast<int64_t>(epa_k) * num_max_dispatch_tokens_per_rank +
                 src_idx) *
                    num_bytes_per_slot);
''')
c = sub(c, '''                        (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank +
                         token_idx) *
                            num_bytes_per_slot);
''', '''                        (static_cast<int64_t>(i) * num_max_dispatch_tokens_per_rank +
                         token_idx) *
                            num_bytes_per_slot);
''')
(HERE / 'epa_combine.cuh').write_text(c)
print('wrote epa_dispatch.cuh, epa_combine.cuh')
