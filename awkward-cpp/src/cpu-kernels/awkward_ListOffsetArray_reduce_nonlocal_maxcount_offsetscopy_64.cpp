// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
  int64_t* maxcount,
  int64_t* offsetscopy,
  const int64_t* offsets,
  int64_t length) {
  *maxcount = 0;
  offsetscopy[0] = offsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    int64_t count = (offsets[i + 1] - offsets[i]);
    if (*maxcount < count) {
      *maxcount = count;
    }
    offsetscopy[i + 1] = offsets[i + 1];
  }
  return success();
}
