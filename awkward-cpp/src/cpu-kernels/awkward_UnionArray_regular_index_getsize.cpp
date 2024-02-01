// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_regular_index_getsize.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_UnionArray_regular_index_getsize(
  int64_t* size,
  const C* fromtags,
  int64_t length) {
  *size = 0;
  for (int64_t i = 0;  i < length;  i++) {
    int64_t tag = (int64_t)fromtags[i];
    if (*size < tag) {
      *size = tag;
    }
  }
  *size = *size + 1;
  return success();
}
ERROR awkward_UnionArray8_regular_index_getsize(
  int64_t* size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index_getsize<int8_t>(
    size,
    fromtags,
    length);
}
