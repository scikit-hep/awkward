// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_flatten_combine.cpp", line)

#include "awkward/kernels.h"

template <typename FROMTAGS,
          typename FROMINDEX,
          typename TOTAGS,
          typename TOINDEX,
          typename T>
ERROR awkward_UnionArray_flatten_combine(
  TOTAGS* totags,
  TOINDEX* toindex,
  T* tooffsets,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length,
  T** offsetsraws) {
  tooffsets[0] = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    FROMTAGS tag = fromtags[i];
    FROMINDEX idx = fromindex[i];
    T start = offsetsraws[tag][idx];
    T stop = offsetsraws[tag][idx + 1];
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
    for (int64_t j = start;  j < stop;  j++) {
      totags[k] = tag;
      toindex[k] = j;
      k++;
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, FROMTAGS, FROMINDEX, TOTAGS, TOINDEX, T) \
  ERROR awkward_UnionArray##SUFFIX(TOTAGS* totags, TOINDEX* toindex, T* tooffsets, const FROMTAGS* fromtags, const FROMINDEX* fromindex, int64_t length, T** offsetsraws) { \
    return awkward_UnionArray_flatten_combine<FROMTAGS, FROMINDEX, TOTAGS, TOINDEX, T>(totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws); \
  }

WRAPPER(32_flatten_combine_64, int8_t, int32_t, int8_t, int64_t, int64_t)
WRAPPER(U32_flatten_combine_64, int8_t, uint32_t, int8_t, int64_t, int64_t)
WRAPPER(64_flatten_combine_64, int8_t, int64_t, int8_t, int64_t, int64_t)
