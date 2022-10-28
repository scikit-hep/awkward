// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique(
  T* toptr,
  int64_t length,
  int64_t* tolength) {

  int64_t j = 0;
  for (int64_t i = 1;  i < length;  i++) {
    if (toptr[j] != toptr[i]) {
      j++;
      toptr[j] = toptr[i];
    }
  }
  *tolength = j + 1;
  return success();
}
ERROR awkward_unique_bool(
  bool* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<bool>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_int8(
  int8_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<int8_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_uint8(
  uint8_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<uint8_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_int16(
  int16_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<int16_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_uint16(
  uint16_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<uint16_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_int32(
  int32_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<int32_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_uint32(
  uint32_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<uint32_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_int64(
  int64_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<int64_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_uint64(
  uint64_t* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<uint64_t>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_float32(
  float* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<float>(
      toptr,
      length,
      tolength);
}
ERROR awkward_unique_float64(
  double* toptr,
  int64_t length,
  int64_t* tolength) {
    return awkward_unique<double>(
      toptr,
      length,
      tolength);
}
