// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_validity.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_validity(
  const C* index,
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

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_IndexedArray##SUFFIX(const C* index, int64_t length, int64_t lencontent, bool isoption) { \
    return awkward_IndexedArray_validity<C>(index, length, lencontent, isoption); \
  }

WRAPPER(32_validity, int32_t)
WRAPPER(U32_validity, uint32_t)
WRAPPER(64_validity, int64_t)
