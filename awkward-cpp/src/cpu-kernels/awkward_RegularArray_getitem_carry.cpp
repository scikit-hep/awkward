// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_carry.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_carry(
  T* tocarry,
  const T* fromcarry,
  int64_t lencarry,
  int64_t size) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      tocarry[i*size + j] = fromcarry[i]*size + j;
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_carry_64(
  int64_t* tocarry,
  const int64_t* fromcarry,
  int64_t lencarry,
  int64_t size) {
  return awkward_RegularArray_getitem_carry<int64_t>(
    tocarry,
    fromcarry,
    lencarry,
    size);
}
