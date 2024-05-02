// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_array_advanced.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_array_advanced(
  T* tocarry,
  T* toadvanced,
  const T* fromadvanced,
  const T* fromarray,
  int64_t length,
  int64_t size) {
  for (int64_t i = 0;  i < length;  i++) {
    tocarry[i] = i*size + fromarray[fromadvanced[i]];
    toadvanced[i] = i;
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_advanced_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromadvanced,
  const int64_t* fromarray,
  int64_t length,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array_advanced<int64_t>(
    tocarry,
    toadvanced,
    fromadvanced,
    fromarray,
    length,
    size);
}
