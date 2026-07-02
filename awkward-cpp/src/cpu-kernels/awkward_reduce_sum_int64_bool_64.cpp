// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_int64_bool_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_reduce_sum_int64_bool_64(
  int64_t* __restrict__ toptr,
  const bool* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t acc = 0;
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      if (fromptr[i]) acc++;
    }
    toptr[bin] = acc;
  }
  return success();
}
