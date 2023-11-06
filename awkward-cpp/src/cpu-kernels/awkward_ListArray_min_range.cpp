// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_min_range.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListArray_min_range(
  int64_t* tomin,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts) {
  int64_t shorter = fromstops[0] - fromstarts[0];
  for (int64_t i = 1;  i < lenstarts;  i++) {
    int64_t rangeval = fromstops[i] - fromstarts[i];
    shorter = (shorter < rangeval) ? shorter : rangeval;
  }
  *tomin = shorter;
  return success();
}
ERROR awkward_ListArray32_min_range(
  int64_t* tomin,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts) {
  return awkward_ListArray_min_range<int32_t>(
    tomin,
    fromstarts,
    fromstops,
    lenstarts);
}
ERROR awkward_ListArrayU32_min_range(
  int64_t* tomin,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts) {
  return awkward_ListArray_min_range<uint32_t>(
    tomin,
    fromstarts,
    fromstops,
    lenstarts);
}
ERROR awkward_ListArray64_min_range(
  int64_t* tomin,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts) {
  return awkward_ListArray_min_range<int64_t>(
    tomin,
    fromstarts,
    fromstops,
    lenstarts);
}
