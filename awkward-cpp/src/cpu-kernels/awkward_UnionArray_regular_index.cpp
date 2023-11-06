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
ERROR awkward_UnionArray8_32_regular_index(
  int32_t* toindex,
  int32_t* current,
  int64_t size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index<int8_t, int32_t>(
    toindex,
    current,
    size,
    fromtags,
    length);
}
ERROR awkward_UnionArray8_U32_regular_index(
  uint32_t* toindex,
  uint32_t* current,
  int64_t size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index<int8_t, uint32_t>(
    toindex,
    current,
    size,
    fromtags,
    length);
}
ERROR awkward_UnionArray8_64_regular_index(
  int64_t* toindex,
  int64_t* current,
  int64_t size,
  const int8_t* fromtags,
  int64_t length) {
  return awkward_UnionArray_regular_index<int8_t, int64_t>(
    toindex,
    current,
    size,
    fromtags,
    length);
}
