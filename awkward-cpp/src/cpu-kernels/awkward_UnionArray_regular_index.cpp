// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_regular_index.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename I>
ERROR awkward_UnionArray_regular_index(
  I* toindex,
  I* current,
  int64_t size,
  const C* fromtags,
  int64_t length) {
  for (int64_t k = 0;  k < size;  k++) {
    current[k] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    C tag = fromtags[i];
    toindex[(size_t)i] = current[(size_t)tag];
    current[(size_t)tag]++;
  }
  return success();
}

#define WRAPPER(SUFFIX, C, I) \
  ERROR awkward_UnionArray##SUFFIX(I* toindex, I* current, int64_t size, const C* fromtags, int64_t length) { \
    return awkward_UnionArray_regular_index<C, I>(toindex, current, size, fromtags, length); \
  }

WRAPPER(64_32_regular_index, int64_t, int32_t)
WRAPPER(64_U32_regular_index, int64_t, uint32_t)
WRAPPER(64_64_regular_index, int64_t, int64_t)
WRAPPER(8_32_regular_index, int8_t, int32_t)
WRAPPER(8_U32_regular_index, int8_t, uint32_t)
WRAPPER(8_64_regular_index, int8_t, int64_t)
