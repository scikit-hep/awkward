// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_nestedfill_tags_index.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename I, typename C>
ERROR awkward_UnionArray_nestedfill_tags_index(
  T* totags,
  I* toindex,
  C* tmpstarts,
  T tag,
  const C* fromcounts,
  int64_t length) {
  I k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    C start = tmpstarts[i];
    C stop = start + fromcounts[i];
    for (int64_t j = start;  j < stop;  j++) {
      totags[j] = tag;
      toindex[j] = k;
      k++;
    }
    tmpstarts[i] = stop;
  }
  return success();
}

#define WRAPPER(SUFFIX, T, I, C) \
  ERROR awkward_UnionArray##SUFFIX(T* totags, I* toindex, C* tmpstarts, T tag, const C* fromcounts, int64_t length) { \
    return awkward_UnionArray_nestedfill_tags_index<T, I, C>(totags, toindex, tmpstarts, tag, fromcounts, length); \
  }

WRAPPER(8_32_nestedfill_tags_index_64, int8_t, int32_t, int64_t)
WRAPPER(8_U32_nestedfill_tags_index_64, int8_t, uint32_t, int64_t)
WRAPPER(8_64_nestedfill_tags_index_64, int8_t, int64_t, int64_t)
WRAPPER(64_64_nestedfill_tags_index_64, int64_t, int64_t, int64_t)
