// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_unique_next_index_and_offsets_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_IndexedArray_unique_next_index_and_offsets_64(
  int64_t* toindex,
  int64_t* tooffsets,
  const int64_t* fromoffsets,
  const int64_t* fromnulls,
  int64_t startslength) {
  int64_t k = 0;
  int64_t ll = 0;
  int64_t shift = 0;
  toindex[0] = ll;
  tooffsets[0] = fromoffsets[0];
  for (int64_t i = 0;  i < startslength;  i++) {
    for (int64_t j = fromoffsets[i]; j < fromoffsets[i + 1]; j++) {
      toindex[k] = ll;
      k += 1;
      ll += 1;
    }
    if (fromnulls[k] == 1) {
      toindex[k] = -1;
      k += 1;
      shift += 1;
    }
    tooffsets[i + 1] = fromoffsets[i + 1] + shift;
  }
  return success();
}
