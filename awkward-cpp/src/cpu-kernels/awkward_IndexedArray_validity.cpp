// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_validity.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_validity(
  const C* __restrict__ index,
  int64_t length,
  int64_t lencontent,
  bool isoption) {
  for (int64_t i = 0;  i < length;  i++) {
    C idx = index[i];
    if (!isoption) {
      if (idx < 0) {
        return failure("index[i] < 0", i, kSliceNone, FILENAME(__LINE__));
      }
    }
    if (idx >= lencontent) {
      return failure("index[i] >= len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  return success();
}

#define WRAPPER(FUNC, C) \
  ERROR FUNC(const C* index, int64_t length, int64_t lencontent, bool isoption) { \
    return awkward_IndexedArray_validity<C>(index, length, lencontent, isoption); \
  }

WRAPPER(awkward_IndexedArray32_validity, int32_t)
WRAPPER(awkward_IndexedArrayU32_validity, uint32_t)
WRAPPER(awkward_IndexedArray64_validity, int64_t)
