// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_range_spreadadvanced.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_range_spreadadvanced(
  T* toadvanced,
  const T* fromadvanced,
  const C* fromoffsets,
  int64_t lenstarts) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    C count = fromoffsets[i + 1] - fromoffsets[i];
    for (int64_t j = 0;  j < count;  j++) {
      toadvanced[fromoffsets[i] + j] = fromadvanced[i];
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, C, T) \
  ERROR awkward_ListArray##SUFFIX(T* toadvanced, const T* fromadvanced, const C* fromoffsets, int64_t lenstarts) { \
    return awkward_ListArray_getitem_next_range_spreadadvanced<C, T>(toadvanced, fromadvanced, fromoffsets, lenstarts); \
  }

WRAPPER(32_getitem_next_range_spreadadvanced_64, int32_t, int64_t)
WRAPPER(U32_getitem_next_range_spreadadvanced_64, uint32_t, int64_t)
WRAPPER(64_getitem_next_range_spreadadvanced_64, int64_t, int64_t)
