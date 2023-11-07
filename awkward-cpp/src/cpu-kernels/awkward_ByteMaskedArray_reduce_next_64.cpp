// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_reduce_next_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ByteMaskedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int8_t* mask,
  const int64_t* parents,
  int64_t length,
  bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == validwhen) {
      nextcarry[k] = i;
      nextparents[k] = parents[i];
      outindex[i] = k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
