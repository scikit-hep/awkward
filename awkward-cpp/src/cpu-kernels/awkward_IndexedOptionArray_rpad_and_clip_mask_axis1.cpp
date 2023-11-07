// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedOptionArray_rpad_and_clip_mask_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1(
  T* toindex,
  const int8_t* frommask,
  int64_t length) {
  int64_t count = 0;
  for (int64_t i = 0; i < length; i++) {
    if (frommask[i]) {
      toindex[i] = -1;
    }
    else {
      toindex[i] = count;
      count++;
    }
  }
  return success();
}
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
  int64_t* toindex,
  const int8_t* frommask,
  int64_t length) {
  return awkward_IndexedOptionArray_rpad_and_clip_mask_axis1<int64_t>(
    toindex,
    frommask,
    length);
}
