// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_mask.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename M>
ERROR awkward_IndexedArray_mask(
  M* tomask,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = (fromindex[i] < 0);
  }
  return success();
}
ERROR awkward_IndexedArray32_mask8(
  int8_t* tomask,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_mask<int32_t, int8_t>(
    tomask,
    fromindex,
    length);
}
ERROR awkward_IndexedArrayU32_mask8(
  int8_t* tomask,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_mask<uint32_t, int8_t>(
    tomask,
    fromindex,
    length);
}
ERROR awkward_IndexedArray64_mask8(
  int8_t* tomask,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_mask<int64_t, int8_t>(
    tomask,
    fromindex,
    length);
}
