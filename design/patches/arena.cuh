// epa::Arena: a VMM-free dynamic memory allocator that GPU kernels call directly.
//
// One ordinary allocation (cudaMalloc, optionally shared with peers through cudaIpc /
// NCCL / NVSHMEM windows) is split into `num_pages` pages of `page_bytes`. Free pages
// live in a lock-free ticket ring, so a kernel can take and give pages with one
// fetch-add per warp, on its own GPU or on a peer's over NVLink, with no host
// involvement (CUDA-graph safe). On top of pages:
//
//   WarpHeap     malloc/free for objects up to a page: a warp owns a page and bumps;
//                any thread on any SM (or peer) frees; a page returns to the ring when
//                its owner retired it and its last object is freed.
//   PagedArray   a growable array of fixed-size rows (append from any thread, any
//                rank): rows are contiguous inside a page, pages are found through a
//                page table. Memory follows the real length, page by page.
//
// Out of memory is a policy chosen at the call site, never a crash:
//   try_*    returns failure (nullptr / -1): the caller keeps the data where it is
//            (e.g. EP rows stay in their producer's Home and are read in place)
//   wait_*   retries until pages are given back (e.g. a ring: the consumer frees)
//
// Layout of the region (offsets from `base`, identical on every rank):
//   Meta            head / tail tickets, free count, stats              256 B
//   slot[N]         u64 per ring slot: sequence << 24 | page id (Vyukov MPMC ring; one
//                   word, so reading a slot is one load, also over NVLink)
//   refs[N]         i32 live objects per page (WarpHeap)
//   data            N pages, aligned to kDataAlign
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

#include <epa/window.cuh>

namespace epa {

struct ArenaMeta {
    unsigned long long head;        // next ring ticket to take (allocations)
    unsigned long long tail;        // next ring ticket to fill (frees)
    int free_pages;                 // pages in the ring that may be taken
    int min_free_pages;             // low-water mark (host reads it to size the arena)
    unsigned long long failed;      // try_* calls that found no pages
    unsigned long long taken;       // pages taken since init
    char pad[256 - 40];
};
static_assert(sizeof(ArenaMeta) == 256, "ArenaMeta is one 256-byte header");

constexpr int64_t kDataAlign = 4096;
constexpr int kOwnerRef = 1 << 30;   // a WarpHeap owner's reference on its current page

struct Arena {
    char* base = nullptr;            // region base (local, or a peer's mapped address)
    int64_t page_bytes = 0;
    int num_pages = 0;
    // Pages start at base + data_off. Default: right after the ring state. A library
    // may put the pages elsewhere in the same allocation (e.g. in memory another phase
    // reuses while the pages are free) and keep the ring state where nothing else writes.
    static constexpr int64_t kDataAfterMeta = INT64_MIN;
    int64_t data_off = kDataAfterMeta;

    // ------------------------------------------------------------------ host side
    static EPA_HD int64_t align_up(int64_t x, int64_t a) { return (x + a - 1) / a * a; }
    static EPA_HD int64_t meta_bytes(int num_pages) {
        return align_up(256 + int64_t(num_pages) * (8 + 4), kDataAlign);
    }
    // Bytes to allocate for `num_pages` pages of `page_bytes`.
    static EPA_HD int64_t bytes_for(int num_pages, int64_t page_bytes) {
        return meta_bytes(num_pages) + int64_t(num_pages) * page_bytes;
    }
    // Pages that fit in `bytes`.
    static EPA_HD int pages_in(int64_t bytes, int64_t page_bytes) {
        int n = static_cast<int>(bytes / page_bytes);
        while (n > 0 and bytes_for(n, page_bytes) > bytes) --n;
        return n;
    }

    EPA_HD ArenaMeta* meta() const { return reinterpret_cast<ArenaMeta*>(base); }
    static constexpr int kIdBits = 24;   // pages per arena < 16 M
    // A slot keeps the low 40 bits of a ticket (the rest of the word is the page id), so
    // tickets are compared modulo 2^40. Safe because at most N < 2^24 tickets share a
    // slot at once; without the mask every take and give spins forever after 2^40 tickets.
    static constexpr unsigned long long kSeqMask = (1ull << (64 - kIdBits)) - 1;
    static EPA_HD unsigned long long seq_of(unsigned long long ticket) { return ticket & kSeqMask; }
    EPA_HD unsigned long long* seq() const { return reinterpret_cast<unsigned long long*>(base + 256); }
    EPA_HD int* refs() const { return reinterpret_cast<int*>(base + 256 + int64_t(num_pages) * 8); }
    EPA_HD char* data() const { return base + (data_off == kDataAfterMeta ? meta_bytes(num_pages) : data_off); }

    // Host: the initial ring state (all pages free), to copy to `base` once.
    static std::vector<char> host_init_image(int num_pages) {
        std::vector<char> img(meta_bytes(num_pages), 0);
        ArenaMeta m{};
        m.head = 0, m.tail = num_pages, m.free_pages = num_pages, m.min_free_pages = num_pages;
        std::memcpy(img.data(), &m, sizeof m);
        auto* slots = reinterpret_cast<unsigned long long*>(img.data() + 256);
        for (int i = 0; i < num_pages; ++i)
            slots[i] = ((static_cast<unsigned long long>(i) + 1) << kIdBits) | static_cast<unsigned>(i);
        return img;
    }
    EPA_HD char* page_ptr(int page) const { return data() + int64_t(page) * page_bytes; }
    EPA_HD int page_of(const void* p) const {
        return static_cast<int>((static_cast<const char*>(p) - data()) / page_bytes);
    }
    EPA_HD bool owns(const void* p) const {
        const char* c = static_cast<const char*>(p);
        return c >= data() and c < data() + int64_t(num_pages) * page_bytes;
    }

#if defined(__CUDACC__)
    // The same arena as mapped on `rank` (its base translated through the peer table).
    __device__ __forceinline__ Arena on(const PeerTable& pt, int rank) const {
        Arena a = *this;
        a.base = pt.peer<char>(base, rank);
        return a;
    }

    template <typename T>
    using sys_ref = cuda::atomic_ref<T, cuda::thread_scope_system>;

    // ------------------------------------------------------------------ page ring
    // Reserve n of the free pages; false if fewer are free (nothing reserved).
    __device__ __forceinline__ bool try_reserve(int n) const {
        // Fast path: one fetch-add, no retry, which matters when the counter is a
        // peer's (every retry is an NVLink round trip). Only near empty do concurrent
        // requests drive it below n; then undo and settle with a few exact CAS tries,
        // so oversubscription does not turn into spurious failures.
        sys_ref<int> f(meta()->free_pages);
        int cur = f.fetch_sub(n, cuda::memory_order_relaxed);
        if (cur < n) {
            f.fetch_add(n, cuda::memory_order_relaxed);
            cur = f.load(cuda::memory_order_relaxed);
            bool ok = false;
            for (int tries = 0; tries < 8 and cur >= n and not ok; ++tries)
                ok = f.compare_exchange_weak(cur, cur - n, cuda::memory_order_relaxed);
            if (not ok) {
                atomicAdd_system(&meta()->failed, 1ull);
                return false;
            }
        }
        atomicMin_system(&meta()->min_free_pages, cur - n);   // no return value: a fire-and-forget reduction
        return true;
    }
    // Ticket of n reserved pages; read them with page_at(ticket, i).
    __device__ __forceinline__ unsigned long long take_reserved(int n) const {
        atomicAdd_system(&meta()->taken, static_cast<unsigned long long>(n));
        return sys_ref<unsigned long long>(meta()->head).fetch_add(n, cuda::memory_order_relaxed);
    }
    // Page id of the i-th page of a ticket (waits for a concurrent give to land).
    __device__ __forceinline__ int page_at(unsigned long long ticket, int i) const {
        const unsigned long long t = ticket + i;
        const auto slot = static_cast<int>(t % num_pages);
        sys_ref<unsigned long long> s(seq()[slot]);
        unsigned long long w;
        while (((w = s.load(cuda::memory_order_acquire)) >> kIdBits) != seq_of(t + 1)) __nanosleep(20);
        s.store((t + num_pages) << kIdBits, cuda::memory_order_release);   // slot free for the give of ticket t + N
        return static_cast<int>(w & ((1u << kIdBits) - 1));
    }
    // One thread: n pages or -1. Its ticket is returned; ids via page_at.
    __device__ __forceinline__ long long try_take(int n) const {
        return try_reserve(n) ? static_cast<long long>(take_reserved(n)) : -1;
    }
    __device__ __forceinline__ unsigned long long wait_take(int n) const {
        for (int backoff = 32;; backoff = backoff < 4096 ? backoff * 2 : backoff) {
            if (try_reserve(n)) return take_reserved(n);
            __nanosleep(backoff);
        }
    }
    // Return n pages (any thread, any rank).
    __device__ __forceinline__ void give(const int* pages, int n) const {
        const unsigned long long t0 = sys_ref<unsigned long long>(meta()->tail).fetch_add(n, cuda::memory_order_relaxed);
        for (int i = 0; i < n; ++i) {
            const unsigned long long t = t0 + i;
            const auto slot = static_cast<int>(t % num_pages);
            sys_ref<unsigned long long> s(seq()[slot]);
            while ((s.load(cuda::memory_order_acquire) >> kIdBits) != seq_of(t)) __nanosleep(20);
            s.store(((t + 1) << kIdBits) | static_cast<unsigned>(pages[i]), cuda::memory_order_release);
        }
        sys_ref<int>(meta()->free_pages).fetch_add(n, cuda::memory_order_release);
    }
    __device__ __forceinline__ void give(int page) const { give(&page, 1); }

    // Whole warp gives n pages: one tail fetch-add, slots filled 32 at a time.
    __device__ __forceinline__ void warp_give(const int* pages, int n) const {
        const unsigned mask = __activemask();
        const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1, width = __popc(mask);
        const int rank_in = __popc(mask & ((1u << lane) - 1));
        unsigned long long t0 = 0;
        if (lane == leader) t0 = sys_ref<unsigned long long>(meta()->tail).fetch_add(n, cuda::memory_order_relaxed);
        t0 = __shfl_sync(mask, t0, leader);
        for (int i = rank_in; i < n; i += width) {
            const unsigned long long t = t0 + i;
            const auto slot = static_cast<int>(t % num_pages);
            sys_ref<unsigned long long> s(seq()[slot]);
            while ((s.load(cuda::memory_order_acquire) >> kIdBits) != seq_of(t)) __nanosleep(20);
            s.store(((t + 1) << kIdBits) | static_cast<unsigned>(pages[i]), cuda::memory_order_release);
        }
        __syncwarp(mask);
        if (lane == leader) sys_ref<int>(meta()->free_pages).fetch_add(n, cuda::memory_order_release);
    }

    // Warp-aggregated take: every active lane asks for n_lane pages of *its* arena
    // (`this` may differ per lane, e.g. one peer per lane); lanes on the same arena
    // share one reservation and one ticket fetch-add. Returns the lane's ticket, or -1
    // if its group's request failed (all-or-nothing per group).
    __device__ __forceinline__ long long warp_try_take(int n_lane) const {
        const unsigned active = __activemask();
        const int lane = threadIdx.x & 31;
        const unsigned group = __match_any_sync(active, reinterpret_cast<unsigned long long>(base));
        const int leader = __ffs(group) - 1;
        int prefix = 0, total = 0;
        for (unsigned m = group; m; m &= m - 1) {
            const int l = __ffs(m) - 1;
            const int v = __shfl_sync(group, n_lane, l);
            if (l < lane) prefix += v;
            total += v;
        }
        long long t = -1;
        if (lane == leader and total > 0) t = try_take(total);
        t = __shfl_sync(group, t, leader);
        return t < 0 or n_lane == 0 ? -1 : t + prefix;
    }
#endif
};

#if defined(__CUDACC__)
// Fill the ring with every page (call once after allocation). A template, so that a
// translation unit that includes this header but never launches it gets no kernel.
template <typename = void>
__global__ void arena_init_kernel(Arena a) {
    const int n = a.num_pages;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        a.seq()[i] = ((static_cast<unsigned long long>(i) + 1) << epa::Arena::kIdBits) | i;   // as if give ticket i wrote it
        a.refs()[i] = 0;
    }
    if (blockIdx.x == 0 and threadIdx.x == 0) {
        ArenaMeta* m = a.meta();
        m->head = 0, m->tail = n, m->free_pages = n, m->min_free_pages = n;
        m->failed = 0, m->taken = 0;
    }
}

// ---------------------------------------------------------------------------------
// WarpHeap: malloc/free for objects of up to one page.
// ---------------------------------------------------------------------------------
// Lives in registers of one warp, replicated in every lane: all 32 lanes call alloc()
// together (a lane that needs nothing passes 0 and gets nullptr). The warp bumps
// inside the page it owns, so allocation needs no atomics except when it takes a new
// page. free() may be called by any thread, on any
// SM or rank: refs[page] counts live objects plus the owner's kOwnerRef, and the
// thread that brings it to zero gives the page back.
struct WarpHeap {
    Arena arena;
    int page = -1;          // owned page (-1: none)
    int64_t offset = 0;     // next free byte in it
    int count = 0;          // objects handed out from it

    __device__ __forceinline__ void retire() {
        if (page < 0) return;
        if ((threadIdx.x & 31) == 0) {
            cuda::atomic_ref<int, cuda::thread_scope_system> r(arena.refs()[page]);
            if (r.fetch_add(count - kOwnerRef, cuda::memory_order_acq_rel) + count - kOwnerRef == 0) arena.give(page);
        }
        page = -1, count = 0;
    }

    // `bytes` <= page_bytes; 16-byte aligned objects. nullptr when the arena is empty
    // (wait = false) or after waiting for a page (wait = true).
    __device__ __forceinline__ void* alloc(int64_t bytes, bool wait = false) {
        constexpr unsigned mask = 0xffffffffu;
        constexpr int leader = 0;
        const int lane = threadIdx.x & 31;
        const int64_t sz = (bytes + 15) / 16 * 16;
        void* result = nullptr;
        unsigned pending = __ballot_sync(mask, sz > 0 and sz <= arena.page_bytes);   // larger: PagedArray
        while (pending) {
            // lanes still pending pack themselves into the current page
            int64_t prefix = 0, total = 0;
            int n_fit = 0;
            for (unsigned m = pending; m; m &= m - 1) {
                const int l = __ffs(m) - 1;
                const int64_t v = __shfl_sync(mask, sz, l);
                if (offset + total + v > arena.page_bytes) break;
                if (l < lane) prefix += v;
                total += v, ++n_fit;
            }
            unsigned fit = 0;
            {
                int k = 0;
                for (unsigned m = pending; m and k < n_fit; m &= m - 1, ++k) fit |= 1u << (__ffs(m) - 1);
            }
            if (page >= 0 and (fit >> lane & 1u)) result = arena.page_ptr(page) + offset + prefix;
            if (page >= 0) offset += total, count += n_fit;
            else fit = 0;
            pending &= ~fit;
            if (pending) {
                // current page full (or none): retire it and take a new one
                retire();
                long long t = -1;
                if (lane == leader) t = wait ? static_cast<long long>(arena.wait_take(1)) : arena.try_take(1);
                t = __shfl_sync(mask, t, leader);
                if (t < 0) break;   // out of memory: pending lanes get nullptr
                int p = 0;
                if (lane == leader) {
                    p = arena.page_at(static_cast<unsigned long long>(t), 0);
                    arena.refs()[p] = kOwnerRef;
                }
                page = __shfl_sync(mask, p, leader), offset = 0, count = 0;
            }
        }
        return result;
    }

    // Any thread, anywhere. `arena` must be the arena that owns p (or its peer view).
    static __device__ __forceinline__ void free(const Arena& arena, void* p) {
        const int pg = arena.page_of(p);
        cuda::atomic_ref<int, cuda::thread_scope_system> r(arena.refs()[pg]);
        if (r.fetch_sub(1, cuda::memory_order_acq_rel) == 1) arena.give(pg);
    }
};

// ---------------------------------------------------------------------------------
// PagedArray: a growable array of fixed-size rows backed by arena pages.
// ---------------------------------------------------------------------------------
// table[max_pages] holds page ids (-1: not yet, -2: arena was empty). append(n) from
// any thread on any rank reserves rows [r, r + n); the thread whose rows include the
// first row of a page takes that page and publishes it. Rows on a -2 page are "not
// placed": the caller keeps them where they are (try policy).
struct PagedArray {
    Arena arena;
    int* table;                       // [max_pages], all -1 when empty
    unsigned long long* length;       // rows appended
    int64_t row_bytes;
    int rows_per_page;
    int max_pages;

    EPA_HD int64_t capacity() const { return int64_t(max_pages) * rows_per_page; }

    // Reserve n rows; returns the first row index (pages are taken here, never later).
    __device__ __forceinline__ long long append(int n, bool wait = false) const {
        const long long r = static_cast<long long>(
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(*length).fetch_add(n, cuda::memory_order_relaxed));
        const long long first_page = (r + rows_per_page - 1) / rows_per_page;   // pages that start inside [r, r+n)
        const long long last_page = (r + n - 1) / rows_per_page;
        const int k = static_cast<int>(last_page - first_page + 1);
        if (k > 0 and first_page < max_pages) {
            const int kk = static_cast<int>((k < max_pages - first_page ? k : max_pages - first_page));
            long long t = wait ? static_cast<long long>(arena.wait_take(kk)) : arena.try_take(kk);
            for (int i = 0; i < kk; ++i) {
                const int id = t < 0 ? -2 : arena.page_at(static_cast<unsigned long long>(t), i);
                cuda::atomic_ref<int, cuda::thread_scope_system>(table[first_page + i]).store(id, cuda::memory_order_release);
            }
        }
        return r;
    }

    // Address of row r, or nullptr if its page could not be taken. Waits while the
    // page is being published by another appender.
    __device__ __forceinline__ char* row(long long r) const {
        const long long p = r / rows_per_page;
        if (p >= max_pages) return nullptr;
        cuda::atomic_ref<int, cuda::thread_scope_system> e(table[p]);
        int id;
        while ((id = e.load(cuda::memory_order_acquire)) == -1) __nanosleep(20);
        return id < 0 ? nullptr : arena.page_ptr(id) + (r % rows_per_page) * row_bytes;
    }

    // Give every page back and empty the array (one thread block; call when no thread
    // is appending or reading). Warps give 32 pages at a time (one tail fetch-add).
    __device__ __forceinline__ void release() const {
        const long long n = static_cast<long long>(*length);
        const long long np = (n + rows_per_page - 1) / rows_per_page;
        const int pages = static_cast<int>(np < max_pages ? np : max_pages);
        const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
        for (int p0 = warp * 32; p0 < pages; p0 += warps * 32) {
            const int cnt = pages - p0 < 32 ? pages - p0 : 32;
            const bool ok = lane >= cnt or table[p0 + lane] >= 0;
            if (__all_sync(0xffffffffu, ok)) {
                arena.warp_give(table + p0, cnt);
            } else if (lane < cnt and table[p0 + lane] >= 0) {
                arena.give(table[p0 + lane]);
            }
            __syncwarp();
            if (lane < cnt) table[p0 + lane] = -1;
        }
        __syncthreads();
        if (threadIdx.x == 0) *length = 0;
    }
};
#endif

}  // namespace epa
