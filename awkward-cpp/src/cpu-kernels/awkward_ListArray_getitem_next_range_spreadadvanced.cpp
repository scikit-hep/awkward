// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_range_spreadadvanced.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_range_spreadadvanced(
  T* __restrict__ toadvanced,
  const T* __restrict__ fromadvanced,
  const C* __restrict__ fromoffsets,
  int64_t lenstarts) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    C count = fromoffsets[i + 1] - fromoffsets[i];
    for (int64_t j = 0;  j < count;  j++) {
      toadvanced[fromoffsets[i] + j] = fromadvanced[i];
    }
  }
  return success();
}

#define WRAPPER(FUNC, C, T) \
  ERROR FUNC(T* toadvanced, const T* fromadvanced, const C* fromoffsets, int64_t lenstarts) { \
    return awkward_ListArray_getitem_next_range_spreadadvanced<C, T>(toadvanced, fromadvanced, fromoffsets, lenstarts); \
  }

WRAPPER(awkward_ListArray32_getitem_next_range_spreadadvanced_64, int32_t, int64_t)
WRAPPER(awkward_ListArrayU32_getitem_next_range_spreadadvanced_64, uint32_t, int64_t)
WRAPPER(awkward_ListArray64_getitem_next_range_spreadadvanced_64, int64_t, int64_t)
