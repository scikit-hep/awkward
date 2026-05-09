// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_rpad_length_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListOffsetArray_rpad_length_axis1(
  C* tooffsets,
  const C* fromoffsets,
  int64_t fromlength,
  int64_t target,
  int64_t* tolength) {
  int64_t length = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0; i < fromlength; i++) {
    int64_t rangeval =
      fromoffsets[i + 1] - fromoffsets[i];
    int64_t longer = (target < rangeval) ? rangeval : target;
    length = length + longer;
    tooffsets[i + 1] = tooffsets[i] + longer;
  }
  *tolength = length;

  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_ListOffsetArray##SUFFIX(C* tooffsets, const C* fromoffsets, int64_t fromlength, int64_t target, int64_t* tolength) { \
    return awkward_ListOffsetArray_rpad_length_axis1<C>(tooffsets, fromoffsets, fromlength, target, tolength); \
  }

WRAPPER(32_rpad_length_axis1, int32_t)
WRAPPER(U32_rpad_length_axis1, uint32_t)
WRAPPER(64_rpad_length_axis1, int64_t)
