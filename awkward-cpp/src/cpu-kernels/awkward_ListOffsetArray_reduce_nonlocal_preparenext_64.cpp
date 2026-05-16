// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_preparenext_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
  int64_t* nextcarry,
  int64_t* nextoffsets,         // length outlength * maxcount + 1
  int64_t /* nextlen */,        // unused; kept in signature for caller compatibility
  int64_t* maxnextparents,
  int64_t* distincts,
  int64_t distinctslen,         // outlength * maxcount
  int64_t* /* offsetscopy */,   // workspace; unused in bin-major layout
  const int64_t* offsets,       // per-row sub-list offsets, length+1 entries
  int64_t /* length */,         // unused; rows are indexed via outer_offsets
  const int64_t* outer_offsets, // outer-bin row offsets, outlength+1 entries
  int64_t outlength,
  int64_t maxcount) {
  // Initialize outputs.
  // -1 sentinel: callers compute the next-layer outlength as
  // `maxnextparents + 1`, so when no elements are processed (empty input)
  // we want that expression to evaluate to 0, not 1.
  *maxnextparents = -1;
  for (int64_t i = 0; i < distinctslen; i++) {
    distincts[i] = -1;
  }

  // For each (outer_bin, col), gather one element from each row in the outer
  // bin whose sub-list extends to that column. The output nextbin index is
  // outer_bin * maxcount + col, and we iterate (outer_bin, col) in order so
  // nextoffsets is monotonically increasing.
  int64_t k = 0;
  nextoffsets[0] = 0;
  for (int64_t outer_bin = 0; outer_bin < outlength; outer_bin++) {
    int64_t row_start = outer_offsets[outer_bin];
    int64_t row_stop = outer_offsets[outer_bin + 1];
    for (int64_t col = 0; col < maxcount; col++) {
      int64_t nextbin = outer_bin * maxcount + col;
      bool any = false;
      for (int64_t row = row_start; row < row_stop; row++) {
        // Does this row have an element at column `col`?
        if (col < offsets[row + 1] - offsets[row]) {
          nextcarry[k] = offsets[row] + col;
          k++;
          any = true;
        }
      }
      nextoffsets[nextbin + 1] = k;
      if (any) {
        // Any non-(-1) value works; downstream only checks for -1.
        if (nextbin < distinctslen) {
          distincts[nextbin] = nextbin;
        }
        if (nextbin > *maxnextparents) {
          *maxnextparents = nextbin;
        }
      }
    }
  }

  return success();
}
