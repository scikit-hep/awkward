// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_overlay_mask.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename M, typename TO>
ERROR awkward_IndexedArray_overlay_mask(
  TO* toindex,
  const M* mask,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    M m = mask[i];
    toindex[i] = (m ? -1 : fromindex[i]);
  }
  return success();
}
ERROR awkward_IndexedArray32_overlay_mask8_to64(
  int64_t* toindex,
  const int8_t* mask,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_overlay_mask<int32_t, int8_t, int64_t>(
    toindex,
    mask,
    fromindex,
    length);
}
ERROR awkward_IndexedArrayU32_overlay_mask8_to64(
  int64_t* toindex,
  const int8_t* mask,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_overlay_mask<uint32_t, int8_t, int64_t>(
    toindex,
    mask,
    fromindex,
    length);
}
ERROR awkward_IndexedArray64_overlay_mask8_to64(
  int64_t* toindex,
  const int8_t* mask,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_overlay_mask<int64_t, int8_t, int64_t>(
    toindex,
    mask,
    fromindex,
    length);
}
