// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_merge_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListOffsetArray_merge_offsets(
  T* tooffsets,
  const int64_t* fromoffsets,
  int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    tooffsets[i] = tooffsets[i] + fromoffsets[i];
  }
  return success();
}
ERROR awkward_ListOffsetArray64_merge_offsets_64(
  int64_t* tooffsets,
  const int64_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_merge_offsets<int64_t>(
    tooffsets,
    fromoffsets,
    length);
}
