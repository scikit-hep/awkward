// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_next_array.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_next_array(
  T* nextcarryptr,
  T* nextadvancedptr,
  const T* carryptr,
  const T* flatheadptr,
  int64_t lencarry,
  int64_t lenflathead,
  int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < lenflathead;  j++) {
      nextcarryptr[i*lenflathead + j] = skip*carryptr[i] + flatheadptr[j];
      nextadvancedptr[i*lenflathead + j] = j;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_array_64(
  int64_t* nextcarryptr,
  int64_t* nextadvancedptr,
  const int64_t* carryptr,
  const int64_t* flatheadptr,
  int64_t lencarry,
  int64_t lenflathead,
  int64_t skip) {
  return awkward_NumpyArray_getitem_next_array(
    nextcarryptr,
    nextadvancedptr,
    carryptr,
    flatheadptr,
    lencarry,
    lenflathead,
    skip);
}
