"""Derive the epa::Arena version of Mooncake's legacy dispatch from the tiered one
(epa_dispatch.cuh, made by make_epa_kernels.py). Run after make_epa_kernels.py.

The receive window disappears. Each (local expert, source) key of a receiver gets
pages of the receiver's arena, rows j of the key at page j / P, slot j % P. Remote
allocation latency (a few NVLink round trips) is hidden by predictive reservation:

  1. at kernel start the sender's count warp takes, per key, the pages the key used
     in the previous call (one reservation per destination per warp) and publishes
     them at once, so data warps send without waiting for the count;
  2. after counting it takes the shortfall (or gives the surplus back) and publishes
     the final page count;
  3. rows beyond the pages it could take stay in the sender's Home (descriptor + pull).

The receiver copies rows into packed_recv_x and gives the pages back, so the arena
holds only rows in flight and is shared by both of Mooncake's buffers.
"""
from pathlib import Path

HERE = Path(__file__).parent


def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f'expected {count} match(es), got {n}:\n{old}'
    return s.replace(old, new)


d = (HERE / 'epa_dispatch.cuh').read_text()
d = sub(d, '#include <epa/window.cuh>\n', '#include <epa/arena.cuh>\n')
d = sub(d, 'namespace mooncake_epa {\n', 'namespace mooncake_arena {\n')
d = sub(d, '}  // namespace mooncake_epa\n', '}  // namespace mooncake_arena\n')

d = sub(d, '''    epa::Window epa_win;   // keys = (local expert, source); keyed = tier-1 rows, shared = pool
    int* epa_recv_desc;
    int* epa_pool_head;
    int* epa_pool_tbl;
    int* epa_pool_base_local;
''', '''    epa::Arena epa_arena;       // this rank's arena (peers: mc_route_put of its base)
    int epa_rows_per_page;
    int epa_max_pages;          // pages per key in the page tables
    int* epa_table;             // receiver: [keys][max_pages] page ids, written by the senders
    int* epa_taken;             // receiver: [keys] pages taken for the key (rows beyond: Home)
    int* epa_table_local;       // sender: [num_experts][max_pages] the same ids, for data warps
    int* epa_avail_local;       // sender: [num_experts] pages usable now | kFinal once counted
    int* epa_pred_local;        // sender: [num_experts] pages the key took last call
    int* epa_recv_desc;
''')
d = sub(d, '''    const epa::Window epa_win = args.epa_win;
    int* epa_recv_desc = args.epa_recv_desc;
    int* epa_pool_head = args.epa_pool_head;
    int* epa_pool_tbl = args.epa_pool_tbl;
    int* epa_pool_base_local = args.epa_pool_base_local;
''', '''    const epa::Arena epa_arena = args.epa_arena;
    const int epa_rows_per_page = args.epa_rows_per_page;
    const int epa_max_pages = args.epa_max_pages;
    int* epa_table = args.epa_table;
    int* epa_taken = args.epa_taken;
    int* epa_table_local = args.epa_table_local;
    int* epa_avail_local = args.epa_avail_local;
    int* epa_pred_local = args.epa_pred_local;
    int* epa_recv_desc = args.epa_recv_desc;
    constexpr int kEpaFinal = 1 << 30;
''')

# sender data warps: wait until the row's page is published, or the count is final
d = sub(d, '''                    // EPA window: keyed row, else shared row, else -1 (stays in Home)
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
                    if (epa_row < 0) {''', '''                    // EPA arena: row slot_idx of the key lives in page slot_idx / P, usable
                    // once published (predicted pages at once, the rest after the count)
                    const int epa_key = dst_expert_local_idx * num_ranks + rank;
                    const int epa_p = slot_idx / epa_rows_per_page;
                    int epa_page = -1;
                    if (lane_id == 0) {
                        cuda::atomic_ref<int, cuda::thread_scope_device> av(epa_avail_local[dst_expert_idx]);
                        int a;
                        while (((a = av.load(cuda::memory_order_acquire)) & ~kEpaFinal) <= epa_p and not (a & kEpaFinal))
                            __nanosleep(32);
                        if ((a & ~kEpaFinal) > epa_p)
                            epa_page = epa_table_local[static_cast<int64_t>(dst_expert_idx) * epa_max_pages + epa_p];
                    }
                    epa_page = __shfl_sync(0xffffffff, epa_page, 0);
                    const int64_t epa_off = epa_page < 0 ? -1 :
                        (epa_arena.page_ptr(epa_page) - epa_arena.base) +
                        static_cast<int64_t>(slot_idx % epa_rows_per_page) * num_bytes_per_msg;
                    if (epa_off < 0) {''')
d = sub(d, '''                    const auto dst_ptr = reinterpret_cast<void*>(
                        reinterpret_cast<uint64_t>(rdma_recv_data_buffer) +
                        epa_row * num_bytes_per_msg);''', '''                    const auto dst_ptr = reinterpret_cast<void*>(epa_arena.base + epa_off);''')

# count warp: predictive take before counting, reconcile after
TAKE = '''
// EPA arena: take n_lane pages of `dst`'s arena for lane key `i`, starting at page
// `first` of the key; ids go to the local and the receiver's page table. Whole warp,
// one reservation per destination; taken_out = n_lane, or 0 if the arena was short.
#define EPA_TAKE_PAGES(n_lane, first, dst, el, i, taken_out)                                              \\
    do {                                                                                                  \\
        epa::Arena da_ = epa_arena;                                                                       \\
        da_.base = static_cast<char*>(mc_route_put(comm_ctx, (dst), epa_arena.base));                     \\
        const long long t_ = da_.warp_try_take(n_lane);                                                   \\
        taken_out = t_ < 0 ? 0 : (n_lane);                                                                \\
        int incl_ = taken_out;                                                                            \\
        for (int o_ = 1; o_ < 32; o_ <<= 1) {                                                             \\
            const int v_ = __shfl_up_sync(0xffffffffu, incl_, o_);                                        \\
            if (lane_id >= o_) incl_ += v_;                                                               \\
        }                                                                                                 \\
        const int tot_ = __shfl_sync(0xffffffffu, incl_, 31), excl_ = incl_ - taken_out;                  \\
        for (int q0_ = 0; q0_ < tot_; q0_ += 32) {                                                        \\
            const int q_ = q0_ + lane_id;                                                                 \\
            int own_ = -1;                                                                                \\
            for (int l_ = 0; l_ < 32; ++l_) {                                                             \\
                const int e_ = __shfl_sync(0xffffffffu, excl_, l_), n_ = __shfl_sync(0xffffffffu, taken_out, l_); \\
                if (q_ < tot_ and e_ <= q_ and q_ < e_ + n_) own_ = l_;                                    \\
            }                                                                                             \\
            const int s_ = own_ < 0 ? 0 : own_;                                                           \\
            const long long ts_ = __shfl_sync(0xffffffffu, t_, s_);                                       \\
            const int es_ = __shfl_sync(0xffffffffu, excl_, s_), ds_ = __shfl_sync(0xffffffffu, (dst), s_); \\
            const int els_ = __shfl_sync(0xffffffffu, (el), s_), is_ = __shfl_sync(0xffffffffu, (i), s_);  \\
            const int fs_ = __shfl_sync(0xffffffffu, (first), s_);                                        \\
            if (own_ >= 0) {                                                                              \\
                epa::Arena a_ = epa_arena;                                                                \\
                a_.base = static_cast<char*>(mc_route_put(comm_ctx, ds_, epa_arena.base));                \\
                const int id_ = a_.page_at(static_cast<unsigned long long>(ts_), q_ - es_);               \\
                const int p_ = fs_ + q_ - es_;                                                            \\
                epa_table_local[static_cast<int64_t>(is_) * epa_max_pages + p_] = id_;                    \\
                *reinterpret_cast<int*>(mc_route_put(comm_ctx, ds_,                                       \\
                    epa_table + static_cast<int64_t>(els_ * num_ranks + rank) * epa_max_pages + p_)) = id_; \\
            }                                                                                             \\
            /* publish pages as they arrive, so a hot key's sends start early */                          \\
            __syncwarp();                                                                                 \\
            __threadfence();                                                                              \\
            const int done_ = min(max(q0_ + 32 - excl_, 0), taken_out);                                   \\
            if (epa_owns and done_ > 0)                                                                   \\
                cuda::atomic_ref<int, cuda::thread_scope_device>(epa_avail_local[(i)]).store(             \\
                    (first) + done_, cuda::memory_order_release);                                         \\
        }                                                                                                 \\
        __syncwarp();                                                                                     \\
    } while (0)
'''
d = sub(d, 'namespace mooncake_arena {\n', 'namespace mooncake_arena {\n' + TAKE)

d = sub(d, '''    if (is_count_worker) {
        EP_DEVICE_ASSERT(num_sms > 1);
''', '''    if (is_count_worker) {
        EP_DEVICE_ASSERT(num_sms > 1);
        // EPA arena, step 1: before counting, take the pages each key used last call
        // and publish them, so data warps send without waiting for the count
        const int epa_nk = min(kNumWarpGroups, num_experts - sm_id * kNumWarpGroups);
        const bool epa_owns = lane_id < epa_nk;
        const int epa_i = sm_id * kNumWarpGroups + lane_id;
        const int epa_dst = epa_owns ? epa_i / num_local_experts : 0;
        const int epa_el = epa_owns ? epa_i % num_local_experts : 0;
        const int epa_pred = epa_owns and active_ranks[epa_dst] ? epa_pred_local[epa_i] : 0;
        int epa_have = 0;
        EPA_TAKE_PAGES(epa_pred, 0, epa_dst, epa_el, epa_i, epa_have);
        __threadfence();
        if (epa_owns)
            cuda::atomic_ref<int, cuda::thread_scope_device>(epa_avail_local[epa_i]).store(epa_have, cuda::memory_order_release);
''')
old_start = '        // EPA tier 2: reserve this rank\'s overflow (count - cap) in the receiver\'s\n'
old_end = '        __syncwarp();\n\n        if (sm_id == 0) {\n'
i0 = d.index(old_start)
i1 = d.index(old_end, i0)
d = d[:i0] + '''        // EPA arena, step 2: reconcile with the count. Take the shortfall, give the
        // surplus back, publish the final page count (rows beyond it stay in Home).
        const bool owns_expert = lane_id < expert_end_idx - expert_begin_idx;
        const int epa_need = epa_owns and active_ranks[epa_dst] ?
            (lane_sum + epa_rows_per_page - 1) / epa_rows_per_page : 0;
        const int epa_more = epa_need > epa_have ? epa_need - epa_have : 0;
        int epa_got = 0;
        EPA_TAKE_PAGES(epa_more, epa_have, epa_dst, epa_el, epa_i, epa_got);
        const int epa_final = epa_need > epa_have ? epa_have + epa_got : epa_need;
        __threadfence();
        if (epa_owns) {
            if (lane_sum > 0)
                *reinterpret_cast<int*>(mc_route_put(comm_ctx, epa_dst, epa_taken + epa_el * num_ranks + rank)) = epa_final;
            cuda::atomic_ref<int, cuda::thread_scope_device>(epa_avail_local[epa_i]).store(
                epa_final | kEpaFinal, cuda::memory_order_release);
            epa_pred_local[epa_i] = epa_need;
        }
        // surplus pages from the prediction go back to the receiver (off the send path)
        if (epa_owns and epa_have > epa_need) {
            epa::Arena a = epa_arena;
            a.base = static_cast<char*>(mc_route_put(comm_ctx, epa_dst, epa_arena.base));
            a.give(epa_table_local + static_cast<int64_t>(epa_i) * epa_max_pages + epa_need, epa_have - epa_need);
        }
''' + d[i1:]
d = sub(d, '        epa_pool_base_local[responsible_expert_idx] = 0;\n',
        '        epa_avail_local[responsible_expert_idx] = 0;\n')

# receiver
d = sub(d, '''            epa_win.row(epa_key, 0) * num_bytes_per_msg;''', '''            0;''')
d = sub(d, '''        // EPA tier 2: where this pair's overflow landed in the pool (-1: tier 3)
        const int epa_over = static_cast<int>(epa_win.overflow(num_recv_tokens));
        const int epa_pool_base = epa_over > 0 ?
            *reinterpret_cast<const volatile int*>(epa_pool_tbl + epa_key) : -1;
        if (epa_stats != nullptr and sub_warp_id == 0 and lane_id == 0 and epa_over > 0)
            atomicAdd(epa_stats + (epa_pool_base >= 0 ? 0 : 1), epa_over);''', '''        // EPA arena: rows of the first `taken` pages are in the arena, the rest in Home
        const int* epa_pages = epa_table + static_cast<int64_t>(epa_key) * epa_max_pages;
        const int epa_taken_pages = num_recv_tokens > 0 ? *reinterpret_cast<const volatile int*>(epa_taken + epa_key) : 0;
        const int epa_in_arena = min(num_recv_tokens, epa_taken_pages * epa_rows_per_page);
        if (epa_stats != nullptr and sub_warp_id == 0 and lane_id == 0 and num_recv_tokens > 0) {
            atomicAdd(epa_stats + 0, epa_in_arena);
            atomicAdd(epa_stats + 1, num_recv_tokens - epa_in_arena);
        }''')
d = sub(d, '''            const int64_t epa_row = epa_win.row(epa_key, i, epa_pool_base);
            if (epa_row >= 0) {
                src_src_idx = reinterpret_cast<int*>(
                    reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) + epa_row * num_bytes_per_msg);''', '''            if (i < epa_in_arena) {
                src_src_idx = reinterpret_cast<int*>(epa_arena.page_ptr(epa_pages[i / epa_rows_per_page]) +
                    static_cast<int64_t>(i % epa_rows_per_page) * num_bytes_per_msg);''')
d = sub(d, '''                (lane_id + 32) < num_scales
                    ? dst_scales[(lane_id + 32) * scale_stride] = scale_1
                    : 0.0f;
            }
        }
    } else {''', '''                (lane_id + 32) < num_scales
                    ? dst_scales[(lane_id + 32) * scale_stride] = scale_1
                    : 0.0f;
            }
        }
        // EPA arena: every row of this key is copied, give its pages back
        mc_bar_sync(warp_group_id + 2, kNumWarpsPerGroup * 32);
        if (sub_warp_id == 0 and epa_taken_pages > 0)
            epa_arena.warp_give(epa_pages, epa_taken_pages);
    } else {''')
for bad in ('epa_win', 'epa_pool', 'epa_status', 'epa_ready'):
    assert bad not in d, [l for l in d.splitlines() if bad in l]
(HERE / 'arena_dispatch.cuh').write_text(d)
print('wrote arena_dispatch.cuh')
