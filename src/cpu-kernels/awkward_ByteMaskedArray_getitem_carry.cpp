// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_getitem_carry.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ByteMaskedArray_getitem_carry(
  int8_t* tomask,
  const int8_t* frommask,
  int64_t lenmask,
  const T* fromcarry,
  int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenmask) {
      return failure("index out of range", i, fromcarry[i], FILENAME(__LINE__));
    }
    tomask[i] = frommask[fromcarry[i]];
  }
  return success();
}
ERROR awkward_ByteMaskedArray_getitem_carry_64(
  int8_t* tomask,
  const int8_t* frommask,
  int64_t lenmask,
  const int64_t* fromcarry,
  int64_t lencarry) {
  return awkward_ByteMaskedArray_getitem_carry(
    tomask,
    frommask,
    lenmask,
    fromcarry,
    lencarry);
}
