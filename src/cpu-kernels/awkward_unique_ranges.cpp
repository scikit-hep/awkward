// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_ranges.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique_ranges(
  T* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
  int64_t m = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    auto start = fromoffsets[i];
    auto stop = fromoffsets[i + 1];
    tooffsets[i] = m;
    toptr[m++] = toptr[start];
    for (int64_t k = start; k < stop; k++) {
      if (toptr[m - 1] != toptr[k]) {
        toptr[m++] = toptr[k];
      }
    }
  }
  tooffsets[offsetslength - 1] = m;
  //
  // std::cout << "offsetslength " << offsetslength << ", length " << length << "\n";
  // int64_t m = 0;
  // int64_t i = 0;
  // int64_t k = 0;
  // tooffsets[k++] = fromoffsets[m];
  // for (; k < offsetslength; k++) {
  //   std::cout << k << ": " << fromoffsets[k] << "(";
  //   if (fromoffsets[k - 1] == fromoffsets[k]) {
  //     tooffsets[k] = m;
  //   }
  //   for (int64_t l = fromoffsets[k - 1]; l < fromoffsets[k]; l++) {
  //     std::cout << "[" << toptr[l] << "], "; i++;
  //     if (toptr[m] != toptr[l]) {
  //       m++;
  //       toptr[m] = toptr[l];
  //       std::cout << toptr[m] << ", ";
  //     }
  //   }
  //   std::cout << ")\n";
  //   tooffsets[k] = m + 1;
  // }
  // std::cout << offsetslength << ": " << length << "(" << m << ", "<< i << ")\n";
  // for (; i < length; i++) {
  //   std::cout << "[" << toptr[i] << "], ";
  //   if (toptr[m] != toptr[i]) {
  //     m++;
  //     toptr[m] = toptr[i];
  //     std::cout << toptr[m] << ", ";
  //   }
  // }
  // std::cout << ")\nk=" << k << "\n";
  // tooffsets[k] = m + 1;
  return success();
}

ERROR awkward_unique_ranges_bool(
  bool* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<bool>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int8(
  int8_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int8_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint8(
  uint8_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint8_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int16(
  int16_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int16_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint16(
  uint16_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint16_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int32(
  int32_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int32_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint32(
  uint32_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint32_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_int64(
  int64_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<int64_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_uint64(
  uint64_t* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<uint64_t>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_float32(
  float* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<float>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}

ERROR awkward_unique_ranges_float64(
  double* toptr,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
    return awkward_unique_ranges<double>(
      toptr,
      length,
      fromoffsets,
      offsetslength,
      tooffsets);
}
