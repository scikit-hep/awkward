// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_getitem_nextcarry_outindex.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ByteMaskedArray_getitem_nextcarry_outindex(
  T* tocarry,
  T* outindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == validwhen) {
      tocarry[k] = i;
      outindex[i] = (T)k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
ERROR awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  int64_t* outindex,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_getitem_nextcarry_outindex<int64_t>(
    tocarry,
    outindex,
    mask,
    length,
    validwhen);
}
