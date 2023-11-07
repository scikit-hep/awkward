// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_int32_bool_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_reduce_sum_int32_bool_64(
  int32_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] += (fromptr[i] != 0);
  }
  return success();
}
