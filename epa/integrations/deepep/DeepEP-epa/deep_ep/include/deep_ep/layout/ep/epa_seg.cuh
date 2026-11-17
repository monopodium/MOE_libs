#pragma once

// EPArena segment (expanded layout, no CPU sync): recv_x is one contiguous region of C rows
// at the tail of the EP buffer, and y is leased into the same rows by the GEMMs. Rows of this
// call beyond C go to ranges borrowed from the unused tail of other GPUs' segments: in
// dispatch's notify block every GPU publishes its own end (epoch-stamped), and a GPU whose
// rows exceed C claims [own_end, C) ranges of the lenders with one CAS each. The copy
// epilogue writes rows >= C there, and combine reads them back through the same table.
//
// The header format and the row lookup belong to EPArena (epa/segment.cuh), so a GEMM can read
// the same rows through epa::RowPull (epa/row_pull.cuh) without knowing DeepEP.
#include <epa/segment.cuh>

namespace deep_ep::layout {

using EPASeg = ::epa::Segment;

}  // namespace deep_ep::layout
