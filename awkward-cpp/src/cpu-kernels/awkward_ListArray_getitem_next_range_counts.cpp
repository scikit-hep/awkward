// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_range_counts.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListArray_getitem_next_range_counts(
  int64_t* total,
  const C* fromoffsets,
  int64_t lenstarts) {
  *total = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    *total = *total + fromoffsets[i + 1] - fromoffsets[i];
  }
  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_ListArray##SUFFIX(int64_t* total, const C* fromoffsets, int64_t lenstarts) { \
    return awkward_ListArray_getitem_next_range_counts<C>(total, fromoffsets, lenstarts); \
  }

WRAPPER(32_getitem_next_range_counts_64, int32_t)
WRAPPER(U32_getitem_next_range_counts_64, uint32_t)
WRAPPER(64_getitem_next_range_counts_64, int64_t)
