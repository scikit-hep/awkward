// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_pad_zero_to_length.cpp", line)

#include "awkward/kernels.h"


template <typename T>
ERROR awkward_NumpyArray_pad_zero_to_length(
  const T* fromptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t target,
  T* toptr) {
  int64_t l_to_char = 0;

  // For each sublist
  for (auto k_sublist = 0; k_sublist < offsetslength - 1; k_sublist++) {
    // Copy from src to dst
    for (int64_t j_from_char = fromoffsets[k_sublist]; j_from_char < fromoffsets[k_sublist + 1]; j_from_char++) {
      toptr[l_to_char++] = fromptr[j_from_char];
    }
    // Pad to remaining width
    auto n_to_pad = target - (fromoffsets[k_sublist + 1] - fromoffsets[k_sublist]);
    for (int64_t j_from_char = 0; j_from_char < n_to_pad; j_from_char++){
      toptr[l_to_char++] = 0;
    }
  }

  return success();
}

ERROR awkward_NumpyArray_pad_zero_to_length_uint8(
  const uint8_t* fromptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t target,
  uint8_t* toptr) {
return awkward_NumpyArray_pad_zero_to_length<uint8_t>(
    fromptr,
    fromoffsets,
    offsetslength,
    target,
    toptr);
}
