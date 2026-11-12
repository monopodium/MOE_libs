#pragma once

// EPArena (Home + Window, epa/window.cuh) for the direct (NVLink) dispatch receive buffer.
//
// Upstream: [num_ranks][num_max_tokens_per_rank] token slots per receiver.
// EPArena:  a window with one key per source rank (keyed = cap rows each, shared =
//           pool rows), the sender's Home rows for tokens without a window row, and
//           small index tables:
//
//   window [R*cap + P] tokens   row = win.row(src, j, base): keyed, then shared
//   home   [T]       tokens   this rank's own tokens, written only for rows left in Home
//   desc   [R][T]    int32    token index in the source's Home
// and, at `tables_offset` (the tail of the EP buffer, which combine never touches):
//   tbl    [R]       int32    pool base of each source (-1: tier 3)
//   head   1         int32    pool head (NVLink fetch-add by sources)
//   pub    [R]       int32    pool base per destination, base + 2 (0 = not ready)
//   last   [R]       int32    pool base per destination of the last non-cached call
//
// Cached dispatches reuse `last` and `tbl`: they must follow a non-cached dispatch
// with the same routing on this buffer.
//
// Arena mode (cap < -1: rows per page P = -cap - 1, pool = number of pages N): the
// receive window becomes an epa::Arena. Its N pages of P tokens sit at the front of the
// buffer (memory combine reuses: pages are free by then); its ring state and these
// tables sit at the tail, after the four tables above:
//   arena  ring state (epa::Arena::meta_bytes(N))
//   rtable [R][maxp] int32   receiver: page ids per source, written by the sources
//   taken  [R]       int32   receiver: pages taken per source (rows beyond: Home)
//   ltable [R][maxp] int32   sender: page ids per destination, written by the destination
//   avail  [R]       int32   sender: pages usable now | kFinal once counted
//   have   [R]       int32   sender: pages held on each destination, written by the destination
//   done   1         int32   (unused)
//   ringb  (shared pool only) ring state of zone B
//   scratch          demand [R] receiver: pages each source needed this call
//                    held   [R] receiver: pages each source holds here (mirror of its `have`)
//                    need   [R] sender: pages needed per destination last call (cached mode)
//
// Page ownership is sticky and receiver-managed: a source keeps its pages on a destination
// across calls. The destination alone touches its ring (GPU scope): after its copy epilogue
// it trims sources holding well over their need and grows sources short of it, writing the
// new page ids into the source's ltable. A sender only reads `have`/ltable and posts
// `taken`/`demand`; rows beyond its held pages stay in its Home for that call.
//
// Shared pool (pool < 0, encoded -(nA << 16 | nB)): one page space at the buffer front holds the
// receive pages, recv_x pages and y pages. Pages are P rows of recv_x (row_bytes each); a receive
// page holds rows_per_page = P - 1 dispatch rows. Zone A = pages [0, nA) covers combine's layout
// (combine overwrites it, so only receive and recv_x pages, dead by then, live there); zone B =
// pages [nA, nA + nB) holds y pages and takes receive / recv_x overflow. Page ids are global.
#include <deep_ep/common/math.cuh>
#include <epa/arena.cuh>
#include <epa/window.cuh>

namespace deep_ep::layout {

struct EPALayout {
    int64_t token_bytes;
    int num_ranks, num_max_tokens, cap, pool;
    int64_t tables_offset;
    epa::Window win;   // keys = source ranks, keyed = cap, shared = pool
    bool arena_mode;
    int rows_per_page, num_pages, max_pages;
    bool shared = false;
    int num_pages_b = 0;
    int64_t row_bytes = 0;
    static constexpr int kFinal = 1 << 30;

    __host__ __device__ static bool is_shared(const int& cap, const int& pool) { return cap < -1 and pool < 0; }
    __host__ __device__ static int zone_a(const int& pool) { return (-pool) >> 16; }
    __host__ __device__ static int zone_b(const int& pool) { return (-pool) & 0xffff; }

    __host__ __device__ EPALayout(const int64_t& token_bytes, const int& num_ranks, const int& num_max_tokens,
                                  const int& cap, const int& pool, const int64_t& tables_offset,
                                  const int64_t& row_bytes = 0) :
        token_bytes(token_bytes), num_ranks(num_ranks), num_max_tokens(num_max_tokens), cap(cap), pool(pool),
        tables_offset(tables_offset), row_bytes(row_bytes) {
        arena_mode = cap < -1;
        shared = is_shared(cap, pool);
        rows_per_page = arena_mode ? -cap - 1 : 0;
        num_pages = arena_mode ? (shared ? zone_a(pool) : pool) : 0;
        num_pages_b = shared ? zone_b(pool) : 0;
        max_pages = arena_mode ? (num_max_tokens + rows_per_page - 1) / rows_per_page : 0;
        win.num_keys = num_ranks, win.keyed = arena_mode ? 0 : cap, win.shared = arena_mode ? 0 : pool;
        win.max_rows = static_cast<int64_t>(num_ranks) * num_max_tokens;
    }

    // Bytes of the tables at `tables_offset` (arena mode: plus ring state and page tables)
    __host__ __device__ static int64_t num_table_bytes(const int& num_ranks, const int& cap = -1, const int& pool = 0,
                                                       const int& num_max_tokens = 0) {
        const int64_t base = 128 + 3 * math::align<int64_t>(num_ranks * 4, 128);
        if (cap >= -1) return base;
        const int64_t maxp = (num_max_tokens + (-cap - 1) - 1) / (-cap - 1);
        const bool sh = is_shared(cap, pool);
        return base + epa::kDataAlign /* alignment slack of arena_offset() */ + epa::Arena::meta_bytes(sh ? zone_a(pool) : pool) +
               2 * math::align<int64_t>(num_ranks * maxp * 4, 128) + 3 * math::align<int64_t>(num_ranks * 4, 128) + 128 +
               scratch_bytes(num_ranks) + (sh ? epa::kDataAlign + epa::Arena::meta_bytes(zone_b(pool)) +
               384 + math::align<int64_t>(zone_b(pool) * 4, 128) : 0);
    }

    // SM 0 scratch: have, want, first, got [R] int32, pfx [R + 1] int32, tickets [R] int64
    __host__ __device__ static int64_t scratch_bytes(const int& num_ranks) {
        return math::align<int64_t>(num_ranks * 8, 128) + math::align<int64_t>(num_ranks * 4 * 4 + (num_ranks + 1) * 4, 128);
    }
    // Shared pool: a page is the recv_x rows that cover rows_per_page dispatch rows
    __host__ __device__ int64_t page_bytes() const {
        const int64_t b = static_cast<int64_t>(rows_per_page) * token_bytes;
        return shared ? (b + row_bytes - 1) / row_bytes * row_bytes : b;
    }
    __host__ __device__ int total_pages() const { return num_pages + num_pages_b; }
    __host__ __device__ int64_t home_offset() const {
        return arena_mode ? static_cast<int64_t>(total_pages()) * page_bytes() : win.rows() * token_bytes;
    }
    __host__ __device__ int64_t desc_offset() const { return home_offset() + static_cast<int64_t>(num_max_tokens) * token_bytes; }
    // End of the per-call data (tier 1, pool, home, descriptors)
    __host__ __device__ int64_t num_data_bytes() const { return desc_offset() + static_cast<int64_t>(num_ranks) * num_max_tokens * 4; }
    __host__ __device__ int64_t tbl_offset() const { return tables_offset; }
    __host__ __device__ int64_t head_offset() const { return tbl_offset() + math::align<int64_t>(num_ranks * 4, 128); }
    __host__ __device__ int64_t pub_offset() const { return head_offset() + 128; }
    __host__ __device__ int64_t last_offset() const { return pub_offset() + math::align<int64_t>(num_ranks * 4, 128); }

    // Typed views into a buffer base
    template <typename T = void>
    __host__ __device__ static T* at(void* base, const int64_t& offset) {
        return reinterpret_cast<T*>(static_cast<int8_t*>(base) + offset);
    }
    // Window row of source `src`'s j-th token given its shared base; nullptr: in Home
    __host__ __device__ void* row(void* base, const int& src, const int& j, const int64_t& shared_base) const {
        const int64_t r = win.row(src, j, shared_base);
        return r < 0 ? nullptr : at(base, r * token_bytes);
    }
    __host__ __device__ void* home(void* base, const int& token) const {
        return at(base, home_offset() + static_cast<int64_t>(token) * token_bytes);
    }
    __host__ __device__ int* desc(void* base, const int& src, const int& j) const {
        return at<int>(base, desc_offset() + (static_cast<int64_t>(src) * num_max_tokens + j) * 4);
    }
    __host__ __device__ int* tbl(void* base) const { return at<int>(base, tbl_offset()); }
    __host__ __device__ int* head(void* base) const { return at<int>(base, head_offset()); }
    __host__ __device__ int* pub(void* base) const { return at<int>(base, pub_offset()); }
    __host__ __device__ int* last(void* base) const { return at<int>(base, last_offset()); }

    // Arena mode
    __host__ __device__ int64_t arena_offset() const {
        return math::align<int64_t>(last_offset() + math::align<int64_t>(num_ranks * 4, 128), epa::kDataAlign);
    }
    __host__ __device__ int64_t rtable_offset() const { return arena_offset() + epa::Arena::meta_bytes(num_pages); }
    __host__ __device__ int64_t table_bytes() const { return math::align<int64_t>(int64_t(num_ranks) * max_pages * 4, 128); }
    __host__ __device__ int64_t taken_offset() const { return rtable_offset() + table_bytes(); }
    __host__ __device__ int64_t ltable_offset() const { return taken_offset() + math::align<int64_t>(num_ranks * 4, 128); }
    __host__ __device__ int64_t avail_offset() const { return ltable_offset() + table_bytes(); }
    __host__ __device__ int64_t pred_offset() const { return avail_offset() + math::align<int64_t>(num_ranks * 4, 128); }
    __host__ __device__ int64_t done_offset() const { return pred_offset() + math::align<int64_t>(num_ranks * 4, 128); }
    // The arena of the buffer at `base` (pages at the buffer front, ring state at the tail)
    __host__ __device__ epa::Arena arena(void* base) const {
        epa::Arena a;
        a.base = at<char>(base, arena_offset());
        a.page_bytes = page_bytes();
        a.num_pages = num_pages;
        a.data_off = -arena_offset();
        return a;
    }
    __host__ __device__ int* rtable(void* base) const { return at<int>(base, rtable_offset()); }
    __host__ __device__ int* taken(void* base) const { return at<int>(base, taken_offset()); }
    __host__ __device__ int* ltable(void* base) const { return at<int>(base, ltable_offset()); }
    __host__ __device__ int* avail(void* base) const { return at<int>(base, avail_offset()); }
    __host__ __device__ int* pred(void* base) const { return at<int>(base, pred_offset()); }
    __host__ __device__ int* have(void* base) const { return at<int>(base, pred_offset()); }
    __host__ __device__ int* done(void* base) const { return at<int>(base, done_offset()); }
    __host__ __device__ long long* scratch_tickets(void* base) const { return at<long long>(base, done_offset() + 128); }
    __host__ __device__ int* scratch_ints(void* base) const {
        return at<int>(base, done_offset() + 128 + math::align<int64_t>(num_ranks * 8, 128));
    }
    // Zone B ring state (shared pool), after the scratch
    __host__ __device__ int64_t ringb_offset() const {
        return math::align<int64_t>(done_offset() + 128 + scratch_bytes(num_ranks), epa::kDataAlign);
    }
    __host__ __device__ epa::Arena arena_b(void* base) const {
        epa::Arena a;
        a.base = at<char>(base, ringb_offset());
        a.page_bytes = page_bytes();
        a.num_pages = num_pages_b;
        a.data_off = -ringb_offset() + static_cast<int64_t>(num_pages) * page_bytes();
        return a;
    }
    // Peer spill (shared pool): after its own takes a GPU lends its free zone-B pages for this call.
    // lend_hdr: [0] epoch (calls published) [1] pages claimed by borrowers [2] pages lent, [4..5] ring ticket (u64), [6] pairs borrowed, [7] claims, [8 + 3j] claim j (lender, first, pairs);
    // lend_pages[nB]: the lent page ids (buffer-global). The lender takes them all back at its next call.
    __host__ __device__ int64_t lend_offset() const {
        return math::align<int64_t>(ringb_offset() + epa::Arena::meta_bytes(num_pages_b), 128);
    }
    __host__ __device__ int* lend_hdr(void* base) const { return at<int>(base, lend_offset()); }
    __host__ __device__ int* lend_pages(void* base) const { return at<int>(base, lend_offset() + 256); }
    __host__ __device__ int* demand(void* base) const { return scratch_ints(base); }
    __host__ __device__ int* held(void* base) const { return scratch_ints(base) + num_ranks; }
    __host__ __device__ int* last_need(void* base) const { return scratch_ints(base) + 2 * num_ranks; }
};

}  // namespace deep_ep::layout
