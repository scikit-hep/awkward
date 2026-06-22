// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_flatten_none2empty.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_IndexedArray_flatten_none2empty(
  T* __restrict__ outoffsets,
  const C* __restrict__ outindex,
  int64_t outindexlength,
  const T* __restrict__ offsets,
  int64_t offsetslength) {
  outoffsets[0] = offsets[0];
  int64_t k = 1;
  for (int64_t i = 0;  i < outindexlength;  i++) {
    C idx = outindex[i];
    if (idx < 0) {
      outoffsets[k] = outoffsets[k - 1];
      k++;
    }
    else if (idx + 1 >= offsetslength) {
      return failure("flattening offset out of range", i, kSliceNone, FILENAME(__LINE__));
    }
    else {
      T count =
        offsets[idx + 1] - offsets[idx];
      outoffsets[k] = outoffsets[k - 1] + count;
      k++;
    }
  }
  return success();
}

#define WRAPPER(FUNC, T, C) \
  ERROR FUNC(T* outoffsets, const C* outindex, int64_t outindexlength, const T* offsets, int64_t offsetslength) { \
    return awkward_IndexedArray_flatten_none2empty<T, C>(outoffsets, outindex, outindexlength, offsets, offsetslength); \
  }

WRAPPER(awkward_IndexedArray32_flatten_none2empty_64, int64_t, int32_t)
WRAPPER(awkward_IndexedArrayU32_flatten_none2empty_64, int64_t, uint32_t)
WRAPPER(awkward_IndexedArray64_flatten_none2empty_64, int64_t, int64_t)
