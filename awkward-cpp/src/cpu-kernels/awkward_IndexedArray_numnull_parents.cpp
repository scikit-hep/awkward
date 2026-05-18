// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_numnull_parents.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_numnull_parents(
  int64_t* numnull,
  int64_t* tolength,
  const C* fromindex,
  int64_t lenindex) {
  *tolength = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    if (fromindex[i] < 0) {
      numnull[i] = 1;
      *tolength = *tolength + 1;
    }
    else {
      numnull[i] = 0;
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_IndexedArray##SUFFIX(int64_t* numnull, int64_t* tolength, const C* fromindex, int64_t lenindex) { \
    return awkward_IndexedArray_numnull_parents<C>(numnull, tolength, fromindex, lenindex); \
  }

WRAPPER(32_numnull_parents, int32_t)
WRAPPER(U32_numnull_parents, uint32_t)
WRAPPER(64_numnull_parents, int64_t)
