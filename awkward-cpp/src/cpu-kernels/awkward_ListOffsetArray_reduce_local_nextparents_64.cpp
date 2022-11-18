// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_local_nextparents_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_local_nextparents_64(
  int64_t* nextparents,
  const int64_t* offsets,
  int64_t length) {
  int64_t initialoffset = offsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = offsets[i] - initialoffset;
         j < offsets[i + 1] - initialoffset;
         j++) {
      nextparents[j] = i;
    }
  }
  return success();
}
