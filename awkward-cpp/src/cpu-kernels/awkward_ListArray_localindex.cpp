// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_localindex.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_localindex(
  T* toindex,
  const C* offsets,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = (int64_t)offsets[i];
    int64_t stop = (int64_t)offsets[i + 1];
    for (int64_t j = start;  j < stop;  j++) {
      toindex[j] = j - start;
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, C, T) \
  ERROR awkward_ListArray##SUFFIX(T* toindex, const C* offsets, int64_t length) { \
    return awkward_ListArray_localindex<C, T>(toindex, offsets, length); \
  }

WRAPPER(32_localindex_64, int32_t, int64_t)
WRAPPER(U32_localindex_64, uint32_t, int64_t)
WRAPPER(64_localindex_64, int64_t, int64_t)
