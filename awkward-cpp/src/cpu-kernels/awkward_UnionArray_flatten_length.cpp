// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_flatten_length.cpp", line)

#include "awkward/kernels.h"

template <typename FROMTAGS, typename FROMINDEX, typename T>
ERROR awkward_UnionArray_flatten_length(
  int64_t* total_length,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length,
  T** offsetsraws) {
  *total_length = 0;
  for (int64_t i = 0;  i < length;  i++) {
    int64_t tag = (int64_t)fromtags[i];
    int64_t idx = (int64_t)fromindex[i];
    T start = offsetsraws[tag][idx];
    T stop = offsetsraws[tag][idx + 1];
    *total_length = *total_length + (int64_t)(stop - start);
  }
  return success();
}
ERROR awkward_UnionArray32_flatten_length_64(
  int64_t* total_length,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_length<int8_t, int32_t, int64_t>(
    total_length,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArrayU32_flatten_length_64(
  int64_t* total_length,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_length<int8_t, uint32_t, int64_t>(
    total_length,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArray64_flatten_length_64(
  int64_t* total_length,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_length<int8_t, int64_t, int64_t>(
    total_length,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
