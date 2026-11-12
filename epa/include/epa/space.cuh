// epa::Space: EPArena v2, cross-GPU memory made transparent (first implementation).
//
// Builds on ArenaGroup (arena.cuh): one Arena per GPU of an NVLink domain.
//
//   Pte          64-bit page table entry: state | moving | owner rank | page | epoch. Objects
//                (recv_x, y, weights, ...) are arrays of Ptes, so a page can move or be lent
//                without the kernel seeing anything but its handle.
//   Board        every GPU's free-page count, bucketed, replicated on every GPU. The owner
//                pushes a new bucket with R remote stores when its count crosses a bucket
//                boundary; borrowers read their local copy (no NVLink round trip).
//   BorrowCache  peer pages taken B at a time (one ticket on the best peer by the Board);
//                later fallbacks pop locally.
//   ReturnBuffer gives of borrowed pages are queued per owner and flushed in parallel
//                (one tail fetch-add per owner), nobody waits on NVLink in give().
//   Phases       set_state() flips an Object's pages at a phase boundary from its masks;
//                lease() lets a guest Object use a host Object's IDLE pages (e.g. y in the
//                recv_x page once GEMM 1 is done). Leases end at a known phase: nothing is
//                given back.
//   Mover        staged pulls: remote tiles are copied into a local staging ring of S pages
//                ahead of the consumer (tiles ordered local first, remote last); the consumer
//                waits on a per-slot ready flag and frees the slot. Eviction / prefetch of
//                PARKED pages use the same copy (move_page).
//
// Layout: every GPU has one region at the same PeerTable offset: [Arena | SpaceState]. All
// arenas have the same page count (ArenaGroup requirement).
#pragma once

#include <epa/arena.cuh>

namespace epa {

// ------------------------------------------------------------------ page table entry
enum PageState : unsigned { kFree = 0, kActive = 1, kIdle = 2, kParked = 3 };

struct Pte {
    // state:3 | moving:1 | owner:8 | page:24 | epoch:28
    static EPA_HD unsigned long long make(unsigned state, int owner, int page, unsigned epoch = 0, bool moving = false) {
        return (static_cast<unsigned long long>(state & 7) << 61) | (static_cast<unsigned long long>(moving) << 60) |
               (static_cast<unsigned long long>(owner & 0xff) << 52) | (static_cast<unsigned long long>(page & 0xffffff) << 28) |
               (epoch & 0xfffffff);
    }
    static EPA_HD unsigned state(unsigned long long e) { return static_cast<unsigned>(e >> 61); }
    static EPA_HD bool moving(unsigned long long e) { return (e >> 60) & 1; }
    static EPA_HD int owner(unsigned long long e) { return static_cast<int>((e >> 52) & 0xff); }
    static EPA_HD int page(unsigned long long e) { return static_cast<int>((e >> 28) & 0xffffff); }
    static EPA_HD int code(unsigned long long e) { return ArenaGroup::code(owner(e), page(e)); }
    static EPA_HD unsigned long long with_state(unsigned long long e, unsigned s) {
        return (e & ~(7ull << 61)) | (static_cast<unsigned long long>(s & 7) << 61);
    }
};

// Masks over the phases of a cycle (bit p = phase p)
struct Object {
    unsigned long long* pte = nullptr;   // [num_vpages], this GPU's memory
    int num_vpages = 0;
    unsigned use_mask = 0, live_mask = 0;
    EPA_HD unsigned state_in(int phase) const {
        return (use_mask >> phase & 1) ? kActive : (live_mask >> phase & 1) ? kParked : kIdle;
    }
};

// Per-GPU state after the arena, same offsets on every GPU
struct SpaceLayout {
    int num_ranks, cache_cap, ret_cap;
    EPA_HD static int64_t al(int64_t x) { return (x + 127) / 128 * 128; }
    EPA_HD int64_t board_off() const { return 0; }                                        // int[R]
    EPA_HD int64_t cache_off() const { return board_off() + al(num_ranks * 4); }          // int[cap]
    EPA_HD int64_t cache_ht_off() const { return cache_off() + al(cache_cap * 4); }       // u64 head | tail << 32, lock
    EPA_HD int64_t ret_off() const { return cache_ht_off() + 128; }                        // int[R][ret_cap]
    EPA_HD int64_t ret_cnt_off() const { return ret_off() + al(int64_t(num_ranks) * ret_cap * 4); }   // int[R]
    EPA_HD int64_t bytes() const { return ret_cnt_off() + al(num_ranks * 4); }
};

struct Space {
    ArenaGroup g;            // local arena + peers (g.pt.local_base = this GPU's region base)
    char* state = nullptr;   // this GPU's SpaceLayout block (inside the region)
    SpaceLayout lay{};
    int board_shift = 0;     // bucket = free_pages >> board_shift
    int batch = 16;          // pages per peer take
    bool use_board = true;   // false: first-fit peers, one page per peer take (ArenaGroup)

    EPA_HD int* board() const { return reinterpret_cast<int*>(state + lay.board_off()); }
    EPA_HD int* cache() const { return reinterpret_cast<int*>(state + lay.cache_off()); }
    EPA_HD unsigned long long* cache_ht() const { return reinterpret_cast<unsigned long long*>(state + lay.cache_ht_off()); }
    EPA_HD int* refill_lock() const { return reinterpret_cast<int*>(state + lay.cache_ht_off() + 8); }
    EPA_HD int* ret() const { return reinterpret_cast<int*>(state + lay.ret_off()); }
    EPA_HD int* ret_cnt() const { return reinterpret_cast<int*>(state + lay.ret_cnt_off()); }

#if defined(__CUDACC__)
    template <typename T> using sys = cuda::atomic_ref<T, cuda::thread_scope_system>;
    template <typename T> using dev = cuda::atomic_ref<T, cuda::thread_scope_device>;

    __device__ __forceinline__ char* ptr(int code) const { return g.ptr(code); }

    // Board: push this GPU's bucket to every GPU when it changed (remote stores, not awaited)
    __device__ __forceinline__ void publish() const {
        const int v = sys<int>(g.local.meta()->free_pages).load(cuda::memory_order_relaxed) >> board_shift;
        int* b = board();
        if (ptx_ld(b + g.rank) == v) return;
        for (int d = 0; d < g.num_ranks; ++d)
            sys<int>(*g.pt.peer<int>(b + g.rank, d)).store(v, cuda::memory_order_relaxed);
    }
    static __device__ __forceinline__ int ptx_ld(const int* p) { return *reinterpret_cast<const volatile int*>(p); }

    // Refill the BorrowCache with `batch` pages from the peer with the most free pages
    // above the floor (Board order, ties from rank + 1); false if no peer has them.
    __device__ __forceinline__ bool refill() const {
        const int R = g.num_ranks;
        bool tried[32] = {};
        for (int round = 1; round < R; ++round) {
            int best = -1, best_v = -1;
            for (int k = 1; k < R; ++k) {
                const int d = (g.rank + k) % R;
                const int v = ptx_ld(board() + d);
                if (not tried[d] and v > best_v) best = d, best_v = v;
            }
            if (best < 0) return false;
            tried[best] = true;
            const Arena a = g.arena(best);
            if (not a.try_reserve(batch, g.peer_floor, false)) continue;
            const unsigned long long t = a.take_reserved(batch);
            atomicAdd_system(&g.local.meta()->borrowed, static_cast<unsigned long long>(batch));
            atomicAdd_system(&a.meta()->lent, static_cast<unsigned long long>(batch));
            const unsigned tail = static_cast<unsigned>(dev<unsigned long long>(*cache_ht()).load(cuda::memory_order_relaxed) >> 32);
            for (int i = 0; i < batch; ++i)
                cache()[(tail + i) % lay.cache_cap] = ArenaGroup::code(best, a.page_at(t, i));
            __threadfence();
            dev<unsigned long long>(*cache_ht()).fetch_add(static_cast<unsigned long long>(batch) << 32, cuda::memory_order_release);
            // the owner's count changed: refresh its bucket on our board from its meta
            board()[best] = sys<int>(a.meta()->free_pages).load(cuda::memory_order_relaxed) >> board_shift;
            return true;
        }
        return false;
    }

    // One page: local ring, then the BorrowCache, then a refill from the best peer.
    // Returns a code, or -1 when no GPU of the group has a page.
    __device__ __forceinline__ int take1(bool allow_local = true) const {
        if (allow_local and g.local.try_reserve(1)) {
            const int p = g.local.page_at(g.local.take_reserved(1), 0);
            publish();
            return ArenaGroup::code(g.rank, p);
        }
        if (not use_board) {   // ArenaGroup baseline: first-fit peers, one page per take
            int owner;
            const long long t = g.try_take(1, &owner);
            return t < 0 ? -1 : g.page_at(owner, static_cast<unsigned long long>(t), 0);
        }
        for (;;) {
            dev<unsigned long long> ht(*cache_ht());
            unsigned long long cur = ht.load(cuda::memory_order_acquire);
            while (static_cast<unsigned>(cur) < static_cast<unsigned>(cur >> 32)) {
                if (ht.compare_exchange_weak(cur, cur + 1, cuda::memory_order_acq_rel))
                    return cache()[static_cast<unsigned>(cur) % lay.cache_cap];
            }
            if (atomicCAS(refill_lock(), 0, 1) == 0) {
                // re-check under the lock: another refill may have landed
                cur = ht.load(cuda::memory_order_acquire);
                const bool ok = static_cast<unsigned>(cur) < static_cast<unsigned>(cur >> 32) or refill();
                atomicExch(refill_lock(), 0);
                if (not ok) return -1;
            } else {
                __nanosleep(64);
            }
        }
    }

    // Whole block: n pages, exact counts, local first then peers in Board order; one ticket
    // per arena, page ids read in parallel. codes[i] = -1 for pages no GPU had.
    // (A notify block that knows its total need uses this instead of n take1 calls.)
    __device__ __forceinline__ int block_take(int n, int* codes, bool allow_local = true, bool allow_peers = true) const {
        __shared__ int pl_owner[32], pl_cnt[32], pl_k;
        __shared__ unsigned long long pl_t[32];
        if (threadIdx.x == 0) {
            int left = n, k = 0;
            const auto grab = [&](const Arena& a, int floor) {
                sys<int> f(a.meta()->free_pages);
                int cur = f.load(cuda::memory_order_relaxed), m;
                while ((m = min(left, cur - floor)) > 0 and not f.compare_exchange_weak(cur, cur - m, cuda::memory_order_relaxed)) {}
                if (m <= 0) return 0;
                atomicMin_system(&a.meta()->min_free_pages, cur - m);
                return m;
            };
            if (allow_local) {
                const int m = grab(g.local, 0);
                if (m > 0) pl_owner[k] = g.rank, pl_cnt[k] = m, pl_t[k] = g.local.take_reserved(m), ++k, left -= m;
            }
            bool tried[32] = {};
            while (allow_peers and left > 0 and k < 32) {
                int best = -1, best_v = -1;
                for (int j = 1; j < g.num_ranks; ++j) {
                    const int d = (g.rank + j) % g.num_ranks;
                    const int v = ptx_ld(board() + d);
                    if (not tried[d] and v > best_v) best = d, best_v = v;
                }
                if (best < 0) break;
                tried[best] = true;
                const Arena a = g.arena(best);
                const int m = grab(a, g.peer_floor);
                if (m > 0) {
                    pl_owner[k] = best, pl_cnt[k] = m, pl_t[k] = a.take_reserved(m), ++k, left -= m;
                    atomicAdd_system(&g.local.meta()->borrowed, static_cast<unsigned long long>(m));
                    atomicAdd_system(&a.meta()->lent, static_cast<unsigned long long>(m));
                }
                board()[best] = sys<int>(a.meta()->free_pages).load(cuda::memory_order_relaxed) >> board_shift;
            }
            if (allow_peers and left > 0 and k < 32 and g.host.base != nullptr) {   // every GPU is out: host memory
                const int m = grab(g.host, 0);
                if (m > 0) {
                    pl_owner[k] = ArenaGroup::kHost, pl_cnt[k] = m, pl_t[k] = g.host.take_reserved(m), ++k, left -= m;
                    atomicAdd_system(&g.local.meta()->borrowed, static_cast<unsigned long long>(m));
                }
            }
            pl_k = k;
        }
        __syncthreads();
        int got = 0;
        for (int j = 0; j < pl_k; ++j) got += pl_cnt[j];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int j = 0, first = 0;
            while (j < pl_k and first + pl_cnt[j] <= i) first += pl_cnt[j], ++j;
            codes[i] = j < pl_k ? ArenaGroup::code(pl_owner[j], g.arena(pl_owner[j]).page_at(pl_t[j], i - first)) : -1;
        }
        __syncthreads();
        if (threadIdx.x == 0 and allow_local) publish();
        return got;
    }

    // Give a page: local -> own ring; borrowed -> owner's ReturnBuffer list (flushed later)
    __device__ __forceinline__ void give(int code) const {
        const int owner = ArenaGroup::owner_of_code(code);
        if (owner == g.rank) {
            g.local.give(ArenaGroup::page_of_code(code));
            publish();
        } else if (not use_board or owner == ArenaGroup::kHost) {
            g.give(code);
        } else {
            const int i = atomicAdd(ret_cnt() + owner, 1);
            ret()[owner * lay.ret_cap + i] = ArenaGroup::page_of_code(code);
        }
    }

    // One block: return every queued borrowed page to its owner, slots filled in parallel
    // (one tail fetch-add per owner). Unused cached pages stay cached (hysteresis); pass
    // drain_cache = true to return them as well.
    __device__ __forceinline__ void flush(bool drain_cache = false) const {
        __shared__ unsigned long long t0;
        __shared__ int n_sh;
        if (drain_cache) {   // move cached pages into the return lists first
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned long long cur = *cache_ht();
                for (unsigned i = static_cast<unsigned>(cur); i < static_cast<unsigned>(cur >> 32); ++i) {
                    const int c = cache()[i % lay.cache_cap];
                    const int o = ArenaGroup::owner_of_code(c);
                    ret()[o * lay.ret_cap + ret_cnt()[o]++] = ArenaGroup::page_of_code(c);
                }
                *cache_ht() = (cur >> 32) | (cur & 0xffffffff00000000ull);
            }
            __syncthreads();
        }
        for (int d = 0; d < g.num_ranks; ++d) {
            if (d == g.rank) continue;
            __syncthreads();
            if (threadIdx.x == 0) {
                n_sh = ret_cnt()[d];
                if (n_sh > 0) t0 = sys<unsigned long long>(g.arena(d).meta()->tail).fetch_add(n_sh, cuda::memory_order_relaxed);
            }
            __syncthreads();
            const int n = n_sh;
            if (n == 0) continue;
            const Arena a = g.arena(d);
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const unsigned long long t = t0 + i;
                sys<unsigned long long> s(a.seq()[static_cast<int>(t % a.num_pages)]);
                while ((s.load(cuda::memory_order_acquire) >> Arena::kIdBits) != Arena::seq_of(t)) __nanosleep(20);
                s.store(((t + 1) << Arena::kIdBits) | static_cast<unsigned>(ret()[d * lay.ret_cap + i]), cuda::memory_order_release);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                sys<int>(a.meta()->free_pages).fetch_add(n, cuda::memory_order_release);
                ret_cnt()[d] = 0;
                board()[d] = sys<int>(a.meta()->free_pages).load(cuda::memory_order_relaxed) >> board_shift;
            }
        }
        __syncthreads();
    }
#endif
};

#if defined(__CUDACC__)
// ------------------------------------------------------------------ phases
// Flip every page of `o` to the state its masks give in `phase` (one grid).
template <typename = void>
__global__ void set_state_kernel(Object o, int phase) {
    const unsigned s = o.state_in(phase);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < o.num_vpages; i += gridDim.x * blockDim.x)
        if (Pte::state(o.pte[i]) != kFree) o.pte[i] = Pte::with_state(o.pte[i], s);
}
// Lease: guest page i takes host page i (the host's pages are IDLE in this phase); the
// lease ends when the host is used again, so nothing is given back.
template <typename = void>
__global__ void lease_kernel(Object guest, Object host, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        guest.pte[i] = Pte::with_state(host.pte[i], kActive);
}

// ------------------------------------------------------------------ Mover
// A copy of `bytes` from src to dst by one block, 16-byte vectors, 4 in flight per thread.
__device__ __forceinline__ void block_copy(char* dst, const char* src, int64_t bytes) {
    const int64_t n = bytes / 16;
    const int4* s = reinterpret_cast<const int4*>(src);
    int4* d = reinterpret_cast<int4*>(dst);
    int64_t i = threadIdx.x;
    for (; i + 3 * blockDim.x < n; i += 4 * blockDim.x) {
        const int4 a = s[i], b = s[i + blockDim.x], c = s[i + 2 * blockDim.x], e = s[i + 3 * blockDim.x];
        d[i] = a, d[i + blockDim.x] = b, d[i + 2 * blockDim.x] = c, d[i + 3 * blockDim.x] = e;
    }
    for (; i < n; i += blockDim.x) d[i] = s[i];
}

// Staging ring: S local pages. Remote tile j goes to slot j % S once the consumer freed the
// slot's previous tile (j - S); ready[slot] = j + 1 when it landed.
struct Staging {
    char* pages;       // [S][page_bytes], local
    int* ready;        // [S]
    int* done;         // [S], tile j + 1 consumed
    int num_slots;
    int64_t page_bytes;
#if defined(__CUDACC__)
    __device__ __forceinline__ char* slot(int j) const { return pages + int64_t(j % num_slots) * page_bytes; }
    __device__ __forceinline__ void wait_ready(int j) const {
        const volatile int* r = ready + j % num_slots;
        while (*r != j + 1) __nanosleep(32);
        __threadfence();
    }
    __device__ __forceinline__ void release(int j) const {
        __threadfence();
        cuda::atomic_ref<int, cuda::thread_scope_device>(done[j % num_slots]).store(j + 1, cuda::memory_order_release);
    }
#endif
};
// Mover kernel: pulls remote tiles src[j] (addresses) into the staging ring in order.
// n_dev: number of tiles, read on the device (known only after the phase's allocation).
template <typename = void>
__global__ void mover_pull_kernel(Staging st, char* const* src, const int* n_dev) {
    const int n = *n_dev;
    for (int j = blockIdx.x; j < n; j += gridDim.x) {
        if (threadIdx.x == 0) {
            const int need = j - st.num_slots + 1;   // the slot's previous tile must be consumed
            if (need > 0) {
                const volatile int* dn = st.done + j % st.num_slots;
                while (*dn < need) __nanosleep(32);
            }
        }
        __syncthreads();
        block_copy(st.slot(j), src[j], st.page_bytes);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            cuda::atomic_ref<int, cuda::thread_scope_device>(st.ready[j % st.num_slots]).store(j + 1, cuda::memory_order_release);
        }
    }
}
// Evict / prefetch: copy pages of an Object to new locations, then publish the new Ptes.
template <typename = void>
__global__ void move_pages_kernel(Space sp, Object o, const int* new_codes, int* old_codes_out) {
    for (int i = blockIdx.x; i < o.num_vpages; i += gridDim.x) {
        const unsigned long long e = o.pte[i];
        const int from = Pte::code(e), to = new_codes[i];
        if (to < 0) {   // no target page: the page stays where it is (still correct, just not moved)
            if (threadIdx.x == 0) old_codes_out[i] = -1;
            continue;
        }
        block_copy(sp.ptr(to), sp.ptr(from), sp.g.local.page_bytes);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            o.pte[i] = Pte::make(Pte::state(e), ArenaGroup::owner_of_code(to), ArenaGroup::page_of_code(to), (e & 0xfffffff) + 1);
            old_codes_out[i] = from;
        }
    }
}
#endif

}  // namespace epa
