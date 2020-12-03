// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_next_at.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_next_at(
  T* nextcarryptr,
  const T* carryptr,
  int64_t lencarry,
  int64_t skip,
  int64_t at) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + at;
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_at_64(
  int64_t* nextcarryptr,
  const int64_t* carryptr,
  int64_t lencarry,
  int64_t skip,
  int64_t at) {
  return awkward_NumpyArray_getitem_next_at(
    nextcarryptr,
    carryptr,
    lencarry,
    skip,
    at);
}
