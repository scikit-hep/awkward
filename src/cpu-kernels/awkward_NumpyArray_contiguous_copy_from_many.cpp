// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_contiguous_copy_from_many.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_contiguous_copy_from_many(
  uint8_t* toptr,
  const uint8_t** fromptrs,
  int64_t* fromlens,
  int64_t len,
  int64_t stride,
  const T* pos) {
  int64_t k = 0;
  int64_t j = 0;
  for (int64_t i = 0;  i < len;  i++) {
    memcpy(&toptr[i*stride], &fromptrs[k][pos[j++]], (size_t)stride);
    if (j >= fromlens[k]) {
      k++;
      j = 0;
    }
  }
  return success();
}
ERROR awkward_NumpyArray_contiguous_copy_from_many_64(
  uint8_t* toptr,
  const uint8_t** fromptrs,
  int64_t* fromlens,
  int64_t len,
  int64_t stride,
  const int64_t* pos) {
  return awkward_NumpyArray_contiguous_copy_from_many<int64_t>(
    toptr,
    fromptrs,
    fromlens,
    len,
    stride,
    pos);
}
