// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_index_rpad_and_clip_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_index_rpad_and_clip_axis1(
  T* tostarts,
  T* tostops,
  int64_t target,
  int64_t length) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    offset = offset + target;
    tostops[i] = offset;
   }
  return success();
}
ERROR awkward_index_rpad_and_clip_axis1_64(
  int64_t* tostarts,
  int64_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_index_rpad_and_clip_axis1<int64_t>(
    tostarts,
    tostops,
    target,
    length);
}
