// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_transform_cumsum.cpp", line)

#include <cmath>

#include "awkward/kernels.h"


template <typename OUT, typename IN>
ERROR awkward_transform_cumsum(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  int64_t offsetslength
) {
  // For each sublist
  for (int64_t i=0; i < offsetslength - 1; i++) {
    int64_t start = offsets[i];
    int64_t stop = offsets[i+1];

    if (start == stop) {
        continue;
    }
    // Within this sublist, for each item
    toptr[start] = fromptr[start];
    for (int64_t j=start + 1; j < stop; j++) {
        toptr[j] = fromptr[j] + toptr[j-1];
    }
  }

  return success();
}
ERROR awkward_transform_cumsum_int64_bool_64(
    int64_t* toptr,
    const bool* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int64_t, bool>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int32_bool_64(
    int32_t* toptr,
    const bool* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int32_t, bool>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int64_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int64_t, int8_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint64_uint8_64(
    uint64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint64_t, uint8_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int64_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int64_t, int16_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint64_uint16_64(
    uint64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint64_t, uint16_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int64_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int64_t, int32_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint64_uint32_64(
    uint64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint64_t, uint32_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int64_t, int64_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint64_uint64_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint64_t, uint64_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_float32_float32_64(
    float* toptr,
    const float* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<float, float>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_float64_float64_64(
    double* toptr,
    const double* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<double, double>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int32_int8_64(
    int32_t* toptr,
    const int8_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int32_t, int8_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint32_uint8_64(
    uint32_t* toptr,
    const uint8_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint32_t, uint8_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int32_int16_64(
    int32_t* toptr,
    const int16_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int32_t, int16_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint32_uint16_64(
    uint32_t* toptr,
    const uint16_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint32_t, uint16_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_int32_int32_64(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<int32_t, int32_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
ERROR awkward_transform_cumsum_uint32_uint32_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength) {
  return awkward_transform_cumsum<uint32_t, uint32_t>(
      toptr,
      fromptr,
      offsets,
      offsetslength);
}
