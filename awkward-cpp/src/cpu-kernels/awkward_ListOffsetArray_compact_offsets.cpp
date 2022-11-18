// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_compact_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListOffsetArray_compact_offsets(
  T* tooffsets,
  const C* fromoffsets,
  int64_t length) {
  int64_t diff = (int64_t)fromoffsets[0];
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[i + 1] = fromoffsets[i + 1] - diff;
  }
  return success();
}
ERROR awkward_ListOffsetArray32_compact_offsets_64(
  int64_t* tooffsets,
  const int32_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_compact_offsets<int32_t, int64_t>(
    tooffsets,
    fromoffsets,
    length);
}
ERROR awkward_ListOffsetArrayU32_compact_offsets_64(
  int64_t* tooffsets,
  const uint32_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_compact_offsets<uint32_t, int64_t>(
    tooffsets,
    fromoffsets,
    length);
}
ERROR awkward_ListOffsetArray64_compact_offsets_64(
  int64_t* tooffsets,
  const int64_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_compact_offsets<int64_t, int64_t>(
    tooffsets,
    fromoffsets,
    length);
}
