// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_cumsum.cpp", line)

#include "awkward/kernels.h"


template <typename T>
ERROR awkward_cumsum(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {

  // For each sublist
  for (int64_t j = 0; j < (offsetslength - 1); j++) {
    T tot = (T)0;
    for (int64_t i = offsets[j];  i < offsets[j+1];  i++) {
        tot += (T)fromptr[i];
        toptr[i] = tot;
    }
  }

  return success();
}

ERROR awkward_cumsum_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_float32(
  float* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_cumsum_float64(
  double* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_cumsum<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
