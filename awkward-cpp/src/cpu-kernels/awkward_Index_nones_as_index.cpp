// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Index_nones_as_index.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_Index_nones_as_index(
  T* toindex,
  int64_t length) {
  int64_t last_index = 0;
  for (int64_t i = 0; i < length; i++) {
    toindex[i] > last_index ? last_index = toindex[i] : last_index;
  }
  for (int64_t i = 0; i < length; i++) {
    toindex[i] == -1 ? toindex[i] = ++last_index : toindex[i];
  }
  return success();
}
ERROR awkward_Index_nones_as_index_64(
  int64_t* toindex,
  int64_t length) {
  return awkward_Index_nones_as_index<int64_t>(
    toindex,
    length);
}
