// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_toRegularArray.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListOffsetArray_toRegularArray(
  int64_t* size,
  const C* fromoffsets,
  int64_t offsetslength) {
  *size = -1;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)fromoffsets[i + 1] - (int64_t)fromoffsets[i];
    if (count < 0) {
      return failure("offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    if (*size == -1) {
      *size = count;
    }
    else if (*size != count) {
      return failure("cannot convert to RegularArray because subarray lengths are not regular", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  if (*size == -1) {
    *size = 0;
  }
  return success();
}
ERROR awkward_ListOffsetArray32_toRegularArray(
  int64_t* size,
  const int32_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_ListOffsetArray_toRegularArray<int32_t>(
    size,
    fromoffsets,
    offsetslength);
}
ERROR awkward_ListOffsetArrayU32_toRegularArray(
  int64_t* size,
  const uint32_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_ListOffsetArray_toRegularArray<uint32_t>(
    size,
    fromoffsets,
    offsetslength);
}
ERROR awkward_ListOffsetArray64_toRegularArray(
  int64_t* size,
  const int64_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_ListOffsetArray_toRegularArray<int64_t>(
    size,
    fromoffsets,
    offsetslength);
}
