// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_reduce_next_fix_offsets_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_IndexedArray_reduce_next_fix_offsets_64(
  int64_t* outoffsets,
  const int64_t* starts,
  int64_t startslength,
  int64_t outindexlength) {
  for (int64_t i = 0;  i < startslength;  i++) {
    outoffsets[i] = starts[i];
  }
  outoffsets[startslength] = outindexlength;
  return success();
}
