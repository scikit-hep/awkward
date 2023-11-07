// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_local_outoffsets_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_local_outoffsets_64(
  int64_t* outoffsets,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    while (last < parents[i]) {
      outoffsets[k] = i;
      k++;
      last++;
    }
  }
  while (k <= outlength) {
    outoffsets[k] = lenparents;
    k++;
  }
  return success();
}
