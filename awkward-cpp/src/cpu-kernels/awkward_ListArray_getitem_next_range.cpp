// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_range.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_range(
  C* tooffsets,
  T* tocarry,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  int64_t k = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                start != kSliceNone, stop != kSliceNone,
                                length);
    if (step > 0) {
      for (int64_t j = regular_start;  j < regular_stop;  j += step) {
        tocarry[k] = fromstarts[i] + j;
        k++;
      }
    }
    else {
      for (int64_t j = regular_start;  j > regular_stop;  j += step) {
        tocarry[k] = fromstarts[i] + j;
        k++;
      }
    }
    tooffsets[i + 1] = (C)k;
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_range_64(
  int32_t* tooffsets,
  int64_t* tocarry,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range<int32_t, int64_t>(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
ERROR awkward_ListArrayU32_getitem_next_range_64(
  uint32_t* tooffsets,
  int64_t* tocarry,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range<uint32_t, int64_t>(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
ERROR awkward_ListArray64_getitem_next_range_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  return awkward_ListArray_getitem_next_range<int64_t, int64_t>(
    tooffsets,
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    start,
    stop,
    step);
}
