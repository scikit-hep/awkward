// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_copy.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique_copy(
  const T* fromptr,
  T* toptr,
  int64_t length,
  int64_t* tolength) {

  int64_t j = 0;
  toptr[0] = fromptr[0];
  for (int64_t i = 1;  i < length;  i++) {
    if (toptr[j] != fromptr[i]) {
      j++;
      toptr[j] = fromptr[i];
    }
  }
  *tolength = j + 1;
  return success();
}
ERROR awkward_unique_copy_bool(
  const bool* fromptr,
  bool* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<bool>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_int8(
  const int8_t* fromptr,
  int8_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<int8_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_uint8(
  const uint8_t* fromptr,
  uint8_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<uint8_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_int16(
  const int16_t* fromptr,
  int16_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<int16_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_uint16(
  const uint16_t* fromptr,
  uint16_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<uint16_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_int32(
  const int32_t* fromptr,
  int32_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<int32_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_uint32(
  const uint32_t* fromptr,
  uint32_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<uint32_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_int64(
  const int64_t* fromptr,
  int64_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<int64_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_uint64(
  const uint64_t* fromptr,
  uint64_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<uint64_t>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_float32(
  const float* fromptr,
  float* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<float>(
      fromptr,
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_copy_float64(
  const double* fromptr,
  double* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique_copy<double>(
      fromptr,
      toptr,
      length,
      tolength);
}
