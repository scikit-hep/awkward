// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_cumprod.cpp", line)

#include "awkward/kernels.h"


template <typename T>
ERROR awkward_cumprod(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {

  // For each sublist
  for (int64_t j = 0; j < (offsetslength - 1); j++) {
    T tot = (T)1;
    for (int64_t i = offsets[j];  i < offsets[j+1];  i++) {
        tot *= (T)fromptr[i];
        toptr[i] = tot;
    }
  }

  return success();
}

ERROR awkward_cumprod_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_float32(
  float* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumprod_float64(
  double* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumprod<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
