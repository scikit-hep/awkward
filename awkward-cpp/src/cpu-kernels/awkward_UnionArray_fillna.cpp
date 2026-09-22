// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_fillna.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_UnionArray_fillna(
  T* __restrict__ toindex,
  const C* __restrict__ fromindex,
  int64_t length) {
  for (int64_t i = 0; i < length; i++)
  {
    toindex[i] = fromindex[i] >= 0 ? fromindex[i] : 0;
  }
  return success();
}

#define WRAPPER(FUNC, T, C) \
  ERROR FUNC(T* toindex, const C* fromindex, int64_t length) { \
    return awkward_UnionArray_fillna<T, C>(toindex, fromindex, length); \
  }

WRAPPER(awkward_UnionArray_fillna_from32_to64, int64_t, int32_t)
WRAPPER(awkward_UnionArray_fillna_fromU32_to64, int64_t, uint32_t)
WRAPPER(awkward_UnionArray_fillna_from64_to64, int64_t, int64_t)
