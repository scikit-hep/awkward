// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_one_mask.cpp", line)

#include "awkward/kernels.h"

template <typename M>
ERROR awkward_one_mask(
  M* tomask,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = 1;
  }
  return success();
}
ERROR awkward_one_mask8(
  int8_t* tomask,
  int64_t length) {
  return awkward_one_mask<int8_t>(tomask, length);
}
