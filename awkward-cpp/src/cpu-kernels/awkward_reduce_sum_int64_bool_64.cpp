// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_int64_bool_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_reduce_sum_int64_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  std::memset(toptr, 0, outlength * sizeof(int64_t));

  for (int64_t i = 0; i < lenparents; i++) {
    toptr[parents[i]] += fromptr[i];
  }
  return success();
}
