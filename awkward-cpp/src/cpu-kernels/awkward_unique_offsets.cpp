// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique_offsets(
  T* tooffsets,
  int64_t length,
  const int64_t* fromoffsets,
  const int64_t* starts,
  int64_t startslength) {
  int64_t j = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[j] = fromoffsets[i];
    for (int64_t k = j; k < startslength - 1; k++) {
      if (starts[j] == starts[j + 1]) {
        tooffsets[j + 1] = fromoffsets[i];
        j++;
      }
    }
    j++;
  }
  tooffsets[startslength] = fromoffsets[length - 1];

  return success();
}

ERROR awkward_unique_offsets_int8(
  int8_t* tooffsets,
  int64_t offsetslength,
  const int64_t* fromoffsets,
  const int64_t* starts,
  int64_t startslength) {
    return awkward_unique_offsets<int8_t>(
      tooffsets,
      offsetslength,
      fromoffsets,
      starts,
      startslength);
}

ERROR awkward_unique_offsets_int16(
  int16_t* tooffsets,
  int64_t offsetslength,
  const int64_t* fromoffsets,
  const int64_t* starts,
  int64_t startslength) {
    return awkward_unique_offsets<int16_t>(
      tooffsets,
      offsetslength,
      fromoffsets,
      starts,
      startslength);
}

ERROR awkward_unique_offsets_int32(
  int32_t* tooffsets,
  int64_t offsetslength,
  const int64_t* fromoffsets,
  const int64_t* starts,
  int64_t startslength) {
    return awkward_unique_offsets<int32_t>(
      tooffsets,
      offsetslength,
      fromoffsets,
      starts,
      startslength);
}

ERROR awkward_unique_offsets_int64(
  int64_t* tooffsets,
  int64_t offsetslength,
  const int64_t* fromoffsets,
  const int64_t* starts,
  int64_t startslength) {
    return awkward_unique_offsets<int64_t>(
      tooffsets,
      offsetslength,
      fromoffsets,
      starts,
      startslength);
}
