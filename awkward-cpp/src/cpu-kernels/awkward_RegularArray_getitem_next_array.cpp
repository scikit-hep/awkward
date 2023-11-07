// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_array.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_array(
  T* tocarry,
  T* toadvanced,
  const T* fromarray,
  int64_t length,
  int64_t lenarray,
  int64_t size) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < lenarray;  j++) {
      tocarry[i*lenarray + j] = i*size + fromarray[j];
      toadvanced[i*lenarray + j] = j;
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromarray,
  int64_t length,
  int64_t lenarray,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array<int64_t>(
    tocarry,
    toadvanced,
    fromarray,
    length,
    lenarray,
    size);
}
