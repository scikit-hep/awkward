// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_reduce_nonlocal_preparenext_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_RegularArray_reduce_nonlocal_preparenext_64(
    int64_t* nextcarry,
    int64_t* nextoffsets,        // length outlength * size + 1
    const int64_t* offsets,      // length outlength + 1
    int64_t size,
    int64_t /* length */,
    int64_t outlength) {
  // For each outer bin and column j in [0, size), we form a "nextbin"
  // numbered nextbin = bin * size + j, populated by one entry per row in the
  // outer bin. By iterating bin outermost and j inside it, the emitted
  // nextoffsets are monotonically increasing — each nextbin is a contiguous
  // run in nextcarry. The (i, j) -> content-index mapping (i * size + j) is
  // unchanged from the parents-based kernel; only the linear ordering
  // differs, which is fine since reductions are order-independent.
  int64_t k = 0;
  nextoffsets[0] = 0;
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t row_start = offsets[bin];
    int64_t row_stop = offsets[bin + 1];
    for (int64_t j = 0; j < size; j++) {
      for (int64_t i = row_start; i < row_stop; i++) {
        nextcarry[k] = i * size + j;
        k++;
      }
      nextoffsets[bin * size + j + 1] = k;
    }
  }
  return success();
}
