// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_reduce_adjust_starts_shifts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_reduce_adjust_starts_shifts_64(
  int64_t* toptr,
  int64_t outlength,
  const int64_t* parents,
  const int64_t* starts,
  const int64_t* shifts) {
  for (int64_t k = 0;  k < outlength;  k++) {
    int64_t i = toptr[k];
    if (i >= 0) {
      int64_t parent = parents[i];
      int64_t start = starts[parent];
      toptr[k] += shifts[i] - start;
    }
  }
  return success();
}
