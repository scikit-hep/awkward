// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_index_rpad_and_clip_axis0.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_index_rpad_and_clip_axis0(
  T* toindex,
  int64_t target,
  int64_t length) {
  int64_t shorter = (target < length ? target : length);
  for (int64_t i = 0; i < shorter; i++) {
    toindex[i] = i;
  }
  for (int64_t i = shorter; i < target; i++) {
    toindex[i] = -1;
  }
  return success();
}
ERROR awkward_index_rpad_and_clip_axis0_64(
  int64_t* toindex,
  int64_t target,
  int64_t length) {
  return awkward_index_rpad_and_clip_axis0<int64_t>(
    toindex,
    target,
    length);
}
