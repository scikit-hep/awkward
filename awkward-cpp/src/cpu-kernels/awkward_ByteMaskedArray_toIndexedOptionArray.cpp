// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_toIndexedOptionArray.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ByteMaskedArray_toIndexedOptionArray(
  T* toindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[i] = ((mask[i] != 0) == validwhen ? i : -1);
  }
  return success();
}
ERROR awkward_ByteMaskedArray_toIndexedOptionArray64(
  int64_t* toindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_toIndexedOptionArray<int64_t>(
    toindex,
    mask,
    length,
    validwhen);
}
