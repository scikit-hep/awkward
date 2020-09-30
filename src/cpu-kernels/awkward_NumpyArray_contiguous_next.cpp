// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_contiguous_next.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_contiguous_next(
  T* topos,
  const T* frompos,
  int64_t length,
  int64_t skip,
  int64_t stride) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < skip;  j++) {
      topos[i*skip + j] = frompos[i] + j*stride;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_contiguous_next_64(
  int64_t* topos,
  const int64_t* frompos,
  int64_t length,
  int64_t skip,
  int64_t stride) {
  return awkward_NumpyArray_contiguous_next<int64_t>(
    topos,
    frompos,
    length,
    skip,
    stride);
}
