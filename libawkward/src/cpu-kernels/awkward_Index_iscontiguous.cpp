// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_regularize_iscontiguous.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_Index_iscontiguous(
  bool* result,
  const T* fromindex,
  int64_t length) {
  *result = true;
  T expecting = 0;
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
  const int8_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<int8_t>(result, fromindex, length);
}
ERROR awkward_IndexU8_iscontiguous(
  bool* result,
  const uint8_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<uint8_t>(result, fromindex, length);
}
ERROR awkward_Index32_iscontiguous(
  bool* result,
  const int32_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<int32_t>(result, fromindex, length);
}
ERROR awkward_IndexU32_iscontiguous(
  bool* result,
  const uint32_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<uint32_t>(result, fromindex, length);
}
ERROR awkward_Index64_iscontiguous(
  bool* result,
  const int64_t* fromindex,
  int64_t length) {
    return awkward_Index_iscontiguous<int64_t>(result, fromindex, length);
}
