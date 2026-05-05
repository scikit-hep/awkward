// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_reduce_adjust_starts_shifts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_reduce_adjust_starts_shifts_64(
  int64_t* toptr,
  int64_t outlength,
  const int64_t* /* offsets */,   // unused: see note in adjust_starts_64
  const int64_t* starts,
  const int64_t* shifts) {
  // See `awkward_NumpyArray_reduce_adjust_starts_64` for the rationale.
  // `parents[toptr[k]] == k` by construction of argmin/argmax, so we use k
  // directly to look up the bin's start.
  for (int64_t k = 0; k < outlength; k++) {
    int64_t i = toptr[k];
    if (i >= 0) {
      toptr[k] += shifts[i] - starts[k];
    }
  }
  return success();
}
