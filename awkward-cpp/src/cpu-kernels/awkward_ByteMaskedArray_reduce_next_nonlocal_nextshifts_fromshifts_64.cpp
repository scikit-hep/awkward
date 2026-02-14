// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const int8_t* mask,
  int64_t length,
  bool valid_when,
  const int64_t* shifts) {
  int64_t nullsum = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == (valid_when != 0)) {
      nextshifts[k] = shifts[i] + nullsum;
      k++;
    }
    else {
      nullsum++;
    }
  }
  return success();
}
