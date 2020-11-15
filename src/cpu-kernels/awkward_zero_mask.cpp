// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_zero_mask.cpp", line)

#include "awkward/kernels.h"

template <typename M>
ERROR awkward_zero_mask(
  M* tomask,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = 0;
  }
  return success();
}
ERROR awkward_zero_mask64(
  int64_t* tomask,
  int64_t length) {
  return awkward_zero_mask<int64_t>(tomask, length);
}
ERROR awkward_zero_mask8(
  int8_t* tomask,
  int64_t length) {
  return awkward_zero_mask<int8_t>(tomask, length);
}
