// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_rpad_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListArray_rpad_axis1(
  T* toindex,
  const C* fromstarts,
  const C* fromstops,
  C* tostarts,
  C* tostops,
  int64_t target,
  int64_t length) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    int64_t rangeval = fromstops[i] - fromstarts[i];
    for (int64_t j = 0; j < rangeval; j++) {
     toindex[offset + j] = fromstarts[i] + j;
    }
    for (int64_t j = rangeval; j < target; j++) {
     toindex[offset + j] = -1;
    }
    offset = (target > rangeval) ? tostarts[i] + target : tostarts[i] + rangeval;
    tostops[i] = offset;
  }
  return success();
}
ERROR awkward_ListArray32_rpad_axis1_64(
  int64_t* toindex,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int32_t* tostarts,
  int32_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_ListArray_rpad_axis1<int64_t, int32_t>(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length);
}
ERROR awkward_ListArrayU32_rpad_axis1_64(
  int64_t* toindex,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  uint32_t* tostarts,
  uint32_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_ListArray_rpad_axis1<int64_t, uint32_t>(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length);
}
ERROR awkward_ListArray64_rpad_axis1_64(
  int64_t* toindex,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t* tostarts,
  int64_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_ListArray_rpad_axis1<int64_t, int64_t>(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length);
}
