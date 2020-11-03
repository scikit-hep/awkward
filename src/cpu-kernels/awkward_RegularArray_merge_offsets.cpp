// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_merge_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_merge_offsets(
  T* tooffsets,
  int64_t tolength,
  int64_t length,
  int64_t size,
  int64_t otherlength,
  int64_t othersize) {
  tooffsets[0] = 0;
  int64_t n = 1;
  for (int64_t i = 1; i < tolength; i++) {
    int64_t tmp = 0;
    if (n * size <= length) {
      tmp += size;
    }
    if (n * othersize <= otherlength) {
      tmp += othersize;
    }
    tmp += tooffsets[i - 1];
    tooffsets[i] = tmp;
    n++;
  }
  return success();
}
ERROR awkward_RegularArray_merge_offsets64(
  int64_t* tooffsets,
  int64_t tolength,
  int64_t length,
  int64_t size,
  int64_t otherlength,
  int64_t othersize) {
  return awkward_RegularArray_merge_offsets<int64_t>(
    tooffsets,
    tolength,
    length,
    size,
    otherlength,
    othersize);
}
