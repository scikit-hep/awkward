// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_mask.cpp", line)

#include "awkward/kernels.h"

template <typename M>
ERROR awkward_ByteMaskedArray_mask(
  M* tomask,
  const M* frommask,
  int64_t length,
  bool validwhen) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = ((frommask[i] != 0) != validwhen);
  }
  return success();
}
ERROR awkward_ByteMaskedArray_mask8(
  int8_t* tomask,
  const int8_t* frommask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_mask(
    tomask,
    frommask,
    length,
    validwhen);
}
