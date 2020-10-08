// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_next_range.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_next_range(
  T* nextcarryptr,
  const T* carryptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenhead;  j++) {
      nextcarryptr[i*lenhead + j] = skip*carryptr[i] + start + j*step;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_range_64(
  int64_t* nextcarryptr,
  const int64_t* carryptr,
  int64_t lencarry,
  int64_t lenhead,
  int64_t skip,
  int64_t start,
  int64_t step) {
  return awkward_NumpyArray_getitem_next_range(
    nextcarryptr,
    carryptr,
    lencarry,
    lenhead,
    skip,
    start,
    step);
}
