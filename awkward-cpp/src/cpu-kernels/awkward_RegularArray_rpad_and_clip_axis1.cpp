// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_rpad_and_clip_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_rpad_and_clip_axis1(
  T* toindex,
  int64_t target,
  int64_t size,
  int64_t length) {
  int64_t shorter = (target < size ? target : size);
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < shorter;  j++) {
      toindex[i*target + j] = i*size + j;
    }
    for (int64_t j = shorter;  j < target;  j++) {
      toindex[i*target + j] = -1;
    }
  }
  return success();
}
ERROR awkward_RegularArray_rpad_and_clip_axis1_64(
  int64_t* toindex,
  int64_t target,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_rpad_and_clip_axis1<int64_t>(
    toindex,
    target,
    size,
    length);
}
