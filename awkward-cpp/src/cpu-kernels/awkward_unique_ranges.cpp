// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_ranges.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique_ranges(
  T* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
  int64_t m = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    tooffsets[i] = m;
    toptr[m++] = toptr[fromoffsets[i]];
    for (int64_t k = fromoffsets[i]; k < fromoffsets[i + 1]; k++) {
      if (toptr[m - 1] != toptr[k]) {
        toptr[m++] = toptr[k];
      }
    }
  }
  tooffsets[offsetslength - 1] = m;

  return success();
}

ERROR awkward_unique_ranges_int8(
  int8_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int8_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint8(
  uint8_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint8_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int16(
  int16_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int16_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint16(
  uint16_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint16_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int32(
  int32_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int32_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint32(
  uint32_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint32_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int64(
  int64_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int64_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint64(
  uint64_t* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint64_t>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_float32(
  float* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<float>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_float64(
  double* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<double>(
      toptr,
      fromoffsets,
      offsetslength,
      tooffsets);
}
