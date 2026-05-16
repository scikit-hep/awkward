// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_reduce_next_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ByteMaskedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextoffsets,   // length outlength + 1
  int64_t* outindex,
  const int8_t* mask,
  const int64_t* offsets,
  int64_t outlength,
  bool validwhen) {
  int64_t k = 0;
  nextoffsets[0] = 0;
  for (int64_t bin = 0; bin < outlength; bin++) {
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      bool is_valid = ((mask[i] != 0) == validwhen);
      if (is_valid) {
        nextcarry[k] = i;
        outindex[i]  = k;
        k++;
      }
      else {
        outindex[i] = -1;
      }
    }
    nextoffsets[bin + 1] = k;
  }
  return success();
}
