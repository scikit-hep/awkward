// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_rpad_and_clip_length_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListArray_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const C* fromstarts,
  const C* fromstops,
  int64_t target,
  int64_t lenstarts) {
  int64_t length = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t rangeval = fromstops[i] - fromstarts[i];
    length += (target > rangeval) ? target : rangeval;
  }
  *tomin = length;
  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_ListArray##SUFFIX(int64_t* tomin, const C* fromstarts, const C* fromstops, int64_t target, int64_t lenstarts) { \
    return awkward_ListArray_rpad_and_clip_length_axis1<C>(tomin, fromstarts, fromstops, target, lenstarts); \
  }

WRAPPER(32_rpad_and_clip_length_axis1, int32_t)
WRAPPER(U32_rpad_and_clip_length_axis1, uint32_t)
WRAPPER(64_rpad_and_clip_length_axis1, int64_t)
