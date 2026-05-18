// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
//
// DEPRECATED — local nextparents builder for RegularArray.
//
// In the migrated pipeline, the next-layer offsets for the local-reduction
// case is just `arange(0, length*size + 1, size)`, returned directly by
// RegularArray._compact_offsets64(True). This kernel is no longer needed.
// Preserved for ABI compatibility; remove once all callers have migrated.

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
