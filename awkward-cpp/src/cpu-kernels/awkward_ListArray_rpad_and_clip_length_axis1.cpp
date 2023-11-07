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
ERROR awkward_ListArray32_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t target,
  int64_t lenstarts) {
  return awkward_ListArray_rpad_and_clip_length_axis1<int32_t>(
    tomin,
    fromstarts,
    fromstops,
    target,
    lenstarts);
}
ERROR awkward_ListArrayU32_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t target,
  int64_t lenstarts) {
  return awkward_ListArray_rpad_and_clip_length_axis1<uint32_t>(
    tomin,
    fromstarts,
    fromstops,
    target,
    lenstarts);
}
ERROR awkward_ListArray64_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t target,
  int64_t lenstarts) {
  return awkward_ListArray_rpad_and_clip_length_axis1<int64_t>(
    tomin,
    fromstarts,
    fromstops,
    target,
    lenstarts);
}
