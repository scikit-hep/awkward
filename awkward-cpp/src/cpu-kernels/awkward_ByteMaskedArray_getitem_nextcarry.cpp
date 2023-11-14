// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_getitem_nextcarry.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ByteMaskedArray_getitem_nextcarry(
  T* tocarry,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == validwhen) {
      tocarry[k] = i;
      k++;
    }
  }
  return success();
}
ERROR awkward_ByteMaskedArray_getitem_nextcarry_64(
  int64_t* tocarry,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_getitem_nextcarry<int64_t>(
    tocarry,
    mask,
    length,
    validwhen);
}
