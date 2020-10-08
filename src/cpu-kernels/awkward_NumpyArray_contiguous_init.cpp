// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_contiguous_init.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_contiguous_init(
  T* toptr,
  int64_t skip,
  int64_t stride) {
  for (int64_t i = 0;  i < skip;  i++) {
    toptr[i] = i*stride;
  }
  return success();
}
ERROR awkward_NumpyArray_contiguous_init_64(
  int64_t* toptr,
  int64_t skip,
  int64_t stride) {
  return awkward_NumpyArray_contiguous_init<int64_t>(
    toptr,
    skip,
    stride);
}
