// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_pad_zero_to_length.cpp", line)

#include "awkward/kernels.h"


template <typename T, typename C>
ERROR awkward_NumpyArray_pad_zero_to_length(
  const T* fromptr,
  const C* fromoffsets,
  int64_t offsetslength,
  int64_t target,
  T* toptr) {
  int64_t l_to_char = 0;

  for (int64_t k_sublist = 0; k_sublist < offsetslength - 1; k_sublist++) {
    int64_t start = fromoffsets[k_sublist];
    int64_t end = fromoffsets[k_sublist + 1];
    int64_t count = end - start;

    int64_t destination_offset = k_sublist * target;

    if (count > 0) {
        std::memcpy(&toptr[destination_offset], &fromptr[start], count * sizeof(T));
    }

    int64_t n_to_pad = target - count;
    if (n_to_pad > 0) {
        std::memset(&toptr[destination_offset + count], 0, n_to_pad * sizeof(T));
    }
}

  return success();
}

ERROR awkward_NumpyArray_pad_zero_to_length_uint8_int32(
  const uint8_t* fromptr,
  const int32_t* fromoffsets,
  int64_t offsetslength,
  int64_t target,
  uint8_t* toptr) {
  return awkward_NumpyArray_pad_zero_to_length<uint8_t, int32_t>(
    fromptr,
    fromoffsets,
    offsetslength,
    target,
    toptr);
}

ERROR awkward_NumpyArray_pad_zero_to_length_uint8_uint32(
  const uint8_t* fromptr,
  const uint32_t* fromoffsets,
  int64_t offsetslength,
  int64_t target,
  uint8_t* toptr) {
  return awkward_NumpyArray_pad_zero_to_length<uint8_t, uint32_t>(
    fromptr,
    fromoffsets,
    offsetslength,
    target,
    toptr);
}

ERROR awkward_NumpyArray_pad_zero_to_length_uint8_int64(
  const uint8_t* fromptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t target,
  uint8_t* toptr) {
  return awkward_NumpyArray_pad_zero_to_length<uint8_t, int64_t>(
    fromptr,
    fromoffsets,
    offsetslength,
    target,
    toptr);
}
