// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_next_range_advanced.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_next_range_advanced(
  T* nextcarryptr,
  T* nextadvancedptr,
  const T* carryptr,
  const T* advancedptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
      nextadvancedptr[i*lenhead + j] = advancedptr[i];
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_range_advanced_64(
  int64_t* nextcarryptr,
  int64_t* nextadvancedptr,
  const int64_t* carryptr,
  const int64_t* advancedptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  return awkward_NumpyArray_getitem_next_range_advanced(
    nextcarryptr,
    nextadvancedptr,
    carryptr,
    advancedptr,
    lencarry,
    lenhead,
    skip,
    start,
    step);
}
