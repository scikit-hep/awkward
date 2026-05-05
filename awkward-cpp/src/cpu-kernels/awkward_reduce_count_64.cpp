// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_count_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_reduce_count_64(
  int64_t* toptr,
  const int64_t* offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    toptr[bin] = offsets[bin + 1] - offsets[bin];
  }
  return success();
}
