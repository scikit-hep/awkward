// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_numnull.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_numnull(
  int64_t* numnull,
  const C* fromindex,
  int64_t lenindex) {
  *numnull = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    if (fromindex[i] < 0) {
      *numnull = *numnull + 1;
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_IndexedArray##SUFFIX(int64_t* numnull, const C* fromindex, int64_t lenindex) { \
    return awkward_IndexedArray_numnull<C>(numnull, fromindex, lenindex); \
  }

WRAPPER(32_numnull, int32_t)
WRAPPER(U32_numnull, uint32_t)
WRAPPER(64_numnull, int64_t)
