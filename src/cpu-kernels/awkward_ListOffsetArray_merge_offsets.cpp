// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_merge_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListOffsetArray_merge_offsets(
  T* tooffsets,
  int64_t length,
  const C* fromleft,
  const int64_t leftlen,
  const C* fromright,
  const int64_t rightlen) {
  T a = 0;
  T b = 0;
  tooffsets[0] = 0;
  for (int64_t i = 1; i < length; i++) {
    if (i < leftlen) {
      a = (T)(fromleft[i] - fromleft[i - 1]);
    } else {
      a = (T)0;
    }
    if (i < rightlen) {
      b = (T)(fromright[i] - fromright[i - 1]);
    } else {
      b = (T)0;
    }
    tooffsets[i] = tooffsets[i - 1] + a + b;
  }
  return success();
}
ERROR awkward_ListOffsetArray32_merge_offsets_64(
  int64_t* tooffsets,
  int64_t length,
  const int32_t* fromleft,
  int64_t leftlen,
  const int32_t* fromright,
  int64_t rightlen) {
  return awkward_ListOffsetArray_merge_offsets<int32_t, int64_t>(
    tooffsets,
    length,
    fromleft,
    leftlen,
    fromright,
    rightlen);
}
ERROR awkward_ListOffsetArrayU32_merge_offsets_64(
  int64_t* tooffsets,
  int64_t length,
  const uint32_t* fromleft,
  int64_t leftlen,
  const uint32_t* fromright,
  int64_t rightlen) {
  return awkward_ListOffsetArray_merge_offsets<uint32_t, int64_t>(
    tooffsets,
    length,
    fromleft,
    leftlen,
    fromright,
    rightlen);
}
ERROR awkward_ListOffsetArray64_merge_offsets_64(
  int64_t* tooffsets,
  int64_t length,
  const int64_t* fromleft,
  int64_t leftlen,
  const int64_t* fromright,
  int64_t rightlen) {
  return awkward_ListOffsetArray_merge_offsets<int64_t, int64_t>(
    tooffsets,
    length,
    fromleft,
    leftlen,
    fromright,
    rightlen);
}
