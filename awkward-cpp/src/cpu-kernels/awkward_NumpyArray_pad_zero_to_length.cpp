// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_pad_zero_to_length.cpp", line)

#include "awkward/kernels.h"


template <typename T>
ERROR awkward_NumpyArray_pad_zero_to_length(
    const T* fromptr,
    const int64_t* fromoffsets,
    int64_t offsetslength,
    int64_t target,
    T* toptr) {
    int64_t l = 0;
    for (auto k = 0; k < offsetslength-1; k++) {
        for (int64_t j=fromoffsets[k]; j<fromoffsets[k+1]; j++) {
            toptr[l++] = fromptr[j];
        }
        auto n_to_pad = target - (fromoffsets[k+1] - fromoffsets[k]);
        for (int64_t j=0; j<n_to_pad; j++){
            toptr[l++] = 0;
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
