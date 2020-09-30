// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_num.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_num(
  T* tonum,
  int64_t size,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tonum[i] = size;
  }
  return success();
}
ERROR awkward_RegularArray_num_64(
  int64_t* tonum,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_num<int64_t>(
    tonum,
    size,
    length);
}
