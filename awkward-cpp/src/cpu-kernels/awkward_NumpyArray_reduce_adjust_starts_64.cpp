// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_reduce_adjust_starts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_reduce_adjust_starts_64(
  int64_t* toptr,
  int64_t outlength,
  const int64_t* offsets,   // unused: see note below
  const int64_t* starts) {
  // For output bin k, toptr[k] is the argmin/argmax result — i.e. the index
  // (within the flat input) of the chosen element. By construction that
  // element belongs to bin k, so the bin lookup `parents[toptr[k]]` would
  // simply yield k. We use k directly and avoid the indirection. The
  // `offsets` argument is kept in the signature for symmetry with the rest
  // of the offsets-pipeline migration; downstream callers may drop it later.
  for (int64_t k = 0; k < outlength; k++) {
    int64_t i = toptr[k];
    if (i >= 0) {
      toptr[k] -= starts[k];
    }
  }
  return success();
}
