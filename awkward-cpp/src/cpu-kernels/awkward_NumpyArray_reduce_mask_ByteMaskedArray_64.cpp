// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_reduce_mask_ByteMaskedArray_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
  int8_t* toptr,
  const int64_t* offsets,
  int64_t outlength) {

  for (int64_t i = 0; i < outlength; i++) {
    // A bin is 'masked' (toptr=1) if the count for that bin is 0.
    // The count for bin i is defined by offsets[i+1] - offsets[i].
    if (offsets[i+1] - offsets[i] > 0) {
      toptr[i] = 0; // Not masked (has data)
    } else {
      toptr[i] = 1; // Masked (empty)
    }
  }
  return success();
}
