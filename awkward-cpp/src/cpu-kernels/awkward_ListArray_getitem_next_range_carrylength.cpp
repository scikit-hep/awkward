// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_range_carrylength.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

template <typename C>
ERROR awkward_ListArray_getitem_next_range_carrylength(
  int64_t* carrylength,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts,
  int64_t start,
  int64_t stop,
  int64_t step) {
  *carrylength = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                  start != kSliceNone, stop != kSliceNone,
                                  length);
    if (step > 0) {
      for (int64_t j = regular_start;  j < regular_stop;  j += step) {
        *carrylength = *carrylength + 1;
      }
    }
    else {
      for (int64_t j = regular_start;  j > regular_stop;  j += step) {
        *carrylength = *carrylength + 1;
      }
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_ListArray##SUFFIX(int64_t* carrylength, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t start, int64_t stop, int64_t step) { \
    return awkward_ListArray_getitem_next_range_carrylength<C>(carrylength, fromstarts, fromstops, lenstarts, start, stop, step); \
  }

WRAPPER(32_getitem_next_range_carrylength, int32_t)
WRAPPER(U32_getitem_next_range_carrylength, uint32_t)
WRAPPER(64_getitem_next_range_carrylength, int64_t)
