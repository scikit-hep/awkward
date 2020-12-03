// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_boolean_nonzero.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_boolean_nonzero(
  T* toptr,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i += stride) {
    if (fromptr[i] != 0) {
      toptr[k] = i;
      k++;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_boolean_nonzero_64(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
  return awkward_NumpyArray_getitem_boolean_nonzero<int64_t>(
    toptr,
    fromptr,
    length,
    stride);
}
