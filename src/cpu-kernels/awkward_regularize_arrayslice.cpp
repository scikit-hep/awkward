// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_regularize_arrayslice.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_regularize_arrayslice(
  T* flatheadptr,
  int64_t lenflathead,
  int64_t length) {
  for (int64_t i = 0;  i < lenflathead;  i++) {
    T original = flatheadptr[i];
    if (flatheadptr[i] < 0) {
      flatheadptr[i] += length;
    }
    if (flatheadptr[i] < 0  ||  flatheadptr[i] >= length) {
      return failure("index out of range", kSliceNone, original, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_regularize_arrayslice_64(
  int64_t* flatheadptr,
  int64_t lenflathead,
  int64_t length) {
  return awkward_regularize_arrayslice<int64_t>(
    flatheadptr,
    lenflathead,
    length);
}
