// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_range.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_range(
  T* tocarry,
  int64_t regular_start,
  int64_t step,
  int64_t length,
  int64_t size,
  int64_t nextsize) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < nextsize;  j++) {
      tocarry[i*nextsize + j] = i*size + regular_start + j*step;
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_range_64(
  int64_t* tocarry,
  int64_t regular_start,
  int64_t step,
  int64_t length,
  int64_t size,
  int64_t nextsize) {
  return awkward_RegularArray_getitem_next_range<int64_t>(
    tocarry,
    regular_start,
    step,
    length,
    size,
    nextsize);
}
