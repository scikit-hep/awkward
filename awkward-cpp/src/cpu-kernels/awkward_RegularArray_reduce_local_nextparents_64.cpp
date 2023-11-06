// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_reduce_local_nextparents_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_RegularArray_reduce_local_nextparents_64(
    int64_t* nextparents,
    int64_t size,
    int64_t length) {
  int64_t k = 0;
  for (int64_t i=0; i < length; i++) {
    for (int64_t j=0; j < size; j++) {
      // We're above the reduction, so choose nextparents such that the
      // reduction result keeps our row structure
      nextparents[k++] = i;
    }
  }
  return success();
}
