// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_range_spreadadvanced.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_range_spreadadvanced(
  T* toadvanced,
  const T* fromadvanced,
  int64_t length,
  int64_t nextsize) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < nextsize;  j++) {
      toadvanced[i*nextsize + j] = fromadvanced[i];
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_range_spreadadvanced_64(
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  int64_t length,
  int64_t nextsize) {
  return awkward_RegularArray_getitem_next_range_spreadadvanced<int64_t>(
    toadvanced,
    fromadvanced,
    length,
    nextsize);
}
