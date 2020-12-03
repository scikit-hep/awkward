// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_next_array_advanced.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_next_array_advanced(
  T* nextcarryptr,
  const T* carryptr,
  const T* advancedptr,
  const T* flatheadptr,
  int64_t lencarry,
  int64_t skip) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    nextcarryptr[i] = skip*carryptr[i] + flatheadptr[advancedptr[i]];
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_array_advanced_64(
  int64_t* nextcarryptr,
  const int64_t* carryptr,
  const int64_t* advancedptr,
  const int64_t* flatheadptr,
  int64_t lencarry,
  int64_t skip) {
  return awkward_NumpyArray_getitem_next_array_advanced(
    nextcarryptr,
    carryptr,
    advancedptr,
    flatheadptr,
    lencarry,
    skip);
}
