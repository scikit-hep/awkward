// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_numnull.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ByteMaskedArray_numnull(
  int64_t* numnull,
  const int8_t* mask,
  int64_t length,
  bool validwhen) {
  *numnull = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) != validwhen) {
      *numnull = *numnull + 1;
    }
  }
  return success();
}
