// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_global_startstop_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_global_startstop_64(
  int64_t* globalstart,
  int64_t* globalstop,
  const int64_t* offsets,
  int64_t length) {
  *globalstart = offsets[0];
  *globalstop = offsets[length];
  return success();
}
