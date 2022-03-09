// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_int64_bool_64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_cumsum_bool(
  T* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  // For each sublist
  for (int64_t j = 0; j < (offsetslength - 1); j++) {
    T tot = 0;
    for (int64_t i = offsets[j];  i < offsets[j+1];  i++) {
        tot += fromptr[i];
        toptr[i] = tot;
    }
  }
  return success();
}
ERROR awkward_cumsum_int32_bool(
  int32_t* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum_bool<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_int64_bool(
  int64_t* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum_bool<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
