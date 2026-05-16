// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64(
  int64_t* nummissing,
  int64_t* missing,
  int64_t* nextshifts,
  const int64_t* offsets,         // per-row sub-list offsets
  int64_t /* length */,           // implied by outer_offsets[outlength]
  const int64_t* /* starts */,    // unused now; first-of-bin detected via outer_offsets
  const int64_t* outer_offsets,   // replaces parents
  int64_t outlength,
  int64_t maxcount,
  int64_t nextlen,
  const int64_t* nextcarry) {
  // For each outer bin, reset the per-column "missing-so-far" counters at
  // the start, then walk rows in that bin. Each row whose sub-list is shorter
  // than maxcount contributes to nummissing[col] for col in [count, maxcount).
  // Column-wise missing counts are written into `missing` so that downstream
  // can index by sub-list element position.
  for (int64_t outer_bin = 0; outer_bin < outlength; outer_bin++) {
    for (int64_t k = 0; k < maxcount; k++) {
      nummissing[k] = 0;
    }
    int64_t row_start = outer_offsets[outer_bin];
    int64_t row_stop = outer_offsets[outer_bin + 1];
    for (int64_t i = row_start; i < row_stop; i++) {
      int64_t start = offsets[i];
      int64_t stop = offsets[i + 1];
      int64_t count = stop - start;

      for (int64_t k = count; k < maxcount; k++) {
        nummissing[k]++;
      }

      for (int64_t j = 0; j < count; j++) {
        missing[start + j] = nummissing[j];
      }
    }
  }

  for (int64_t j = 0; j < nextlen; j++) {
    nextshifts[j] = missing[nextcarry[j]];
  }
  return success();
}
