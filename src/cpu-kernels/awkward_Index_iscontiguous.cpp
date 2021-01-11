// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_regularize_arrayslice.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_Index_iscontiguous(
  bool* result,
  int64_t low,
  int64_t high,
  const T* fromindex,
  int64_t length) {
  *result = true;
  T expecting = (T)low;
  for (int64_t i = 0;  i < length;  i++) {
    if (fromindex[i] != expecting) {
      *result = false;
      return success();
    }
    expecting++;
  }
  return success();
}
ERROR awkward_Index8_iscontiguous(
  bool* result,
  int64_t low,
  int64_t high,
  const int8_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<int8_t>(result, low, high, fromindex, length);
}
ERROR awkward_IndexU8_iscontiguous(
  bool* result,
  int64_t low,
  int64_t high,
  const uint8_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<uint8_t>(result, low, high, fromindex, length);
}
ERROR awkward_Index32_iscontiguous(
  bool* result,
  int64_t low,
  int64_t high,
  const int32_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<int32_t>(result, low, high, fromindex, length);
}
ERROR awkward_IndexU32_iscontiguous(
  bool* result,
  int64_t low,
  int64_t high,
  const uint32_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<uint32_t>(result, low, high, fromindex, length);
}
ERROR awkward_Index64_iscontiguous(
  bool* result,
  int64_t low,
  int64_t high,
  const int64_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<int64_t>(result, low, high, fromindex, length);
}
