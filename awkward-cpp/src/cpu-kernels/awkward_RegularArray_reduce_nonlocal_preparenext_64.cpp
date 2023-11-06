// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_reduce_nonlocal_preparenext_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_RegularArray_reduce_nonlocal_preparenext_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    const int64_t* parents,
    int64_t size,
    int64_t length) {
  int64_t k = 0;
  for (int64_t j = 0; j < size; j++) {
    for (int64_t i = 0; i < length; i++) {
        // nextparents needs to be locally contiguous so order the output
        // by the transpose, i.e. ensure that nextparents is sorted
        nextcarry[k] = i * size + j;
        nextparents[k] = parents[i] * size + j;
        k++;

    }
  }
  return success();
}
