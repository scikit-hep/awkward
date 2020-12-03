// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_findgaps_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
  int64_t* gaps,
  const int64_t* parents,
  int64_t lenparents) {
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (last < parent) {
      gaps[k] = parent - last;
      k++;
      last = parent;
    }
  }
  return success();
}
