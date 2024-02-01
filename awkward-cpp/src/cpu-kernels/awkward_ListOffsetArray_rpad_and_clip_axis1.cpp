// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_rpad_and_clip_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_and_clip_axis1(
  T* toindex,
  const C* fromoffsets,
  int64_t length,
  int64_t target) {
  for (int64_t i = 0; i < length; i++) {
    int64_t rangeval = (T)(fromoffsets[i + 1] - fromoffsets[i]);
    int64_t shorter = (target < rangeval) ? target : rangeval;
    for (int64_t j = 0; j < shorter; j++) {
      toindex[i*target + j] = (T)fromoffsets[i] + j;
    }
    for (int64_t j = shorter; j < target; j++) {
      toindex[i*target + j] = -1;
    }
  }
  return success();
}
ERROR awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
  int64_t* toindex,
  const int32_t* fromoffsets,
  int64_t length,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int32_t>(
    toindex,
    fromoffsets,
    length,
    target);
}
ERROR awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
  int64_t* toindex,
  const uint32_t* fromoffsets,
  int64_t length,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, uint32_t>(
    toindex,
    fromoffsets,
    length,
    target);
}
ERROR awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
  int64_t* toindex,
  const int64_t* fromoffsets,
  int64_t length,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int64_t>(
    toindex,
    fromoffsets,
    length,
    target);
}
