// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_unique_ranges.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_unique_ranges(
    T* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {

  int64_t slen = 0;
  int64_t index = 0;
  int64_t counter = 0;
  int64_t start = 0;
  int64_t k = 0;
  bool differ = false;
  outoffsets[counter++] = offsets[0];
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    differ = false;
    if (offsets[i + 1] - offsets[i] != slen  &&  offsets[i + 1] - offsets[i] > 0) {
      differ = true;
    }
    else {
      k = 0;
      for (int64_t j = offsets[i]; j < offsets[i + 1]; j++) {
        if (toptr[start + k++] != toptr[j]) {
          differ = true;
        }
      }
    }
    if (differ) {
      for (int64_t j = offsets[i]; j < offsets[i + 1]; j++) {
        toptr[index++] = toptr[j];
        start = offsets[i];
     }
     outoffsets[counter++] = index;
   }
   slen = offsets[i + 1] - offsets[i];
  }
  *tolength = counter;

  return success();
}

ERROR awkward_NumpyArray_unique_ranges_int8(
    int8_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<int8_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_uint8(
    uint8_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<uint8_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_int16(
    int16_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<int16_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_uint16(
    uint16_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<uint16_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_int32(
    int32_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<int32_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_uint32(
    uint32_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<uint32_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_int64(
    int64_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<int64_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_uint64(
    uint64_t* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<uint64_t>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_float32(
    float* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<float>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
ERROR awkward_NumpyArray_unique_ranges_float64(
    double* toptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    int64_t* tolength) {
      return awkward_NumpyArray_unique_ranges<double>(
        toptr,
        offsets,
        offsetslength,
        outoffsets,
        tolength);
}
