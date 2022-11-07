// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_flatten_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListOffsetArray_flatten_offsets(
  T* tooffsets,
  const C* outeroffsets,
  int64_t outeroffsetslen,
  const T* inneroffsets,
  int64_t /* inneroffsetslen */) {  // FIXME: this argument is not needed
  for (int64_t i = 0;  i < outeroffsetslen;  i++) {
    tooffsets[i] =
      inneroffsets[outeroffsets[i]];
  }
  return success();
}
ERROR awkward_ListOffsetArray32_flatten_offsets_64(
  int64_t* tooffsets,
  const int32_t* outeroffsets,
  int64_t outeroffsetslen,
  const int64_t* inneroffsets,
  int64_t inneroffsetslen) {
  return awkward_ListOffsetArray_flatten_offsets<int64_t, int32_t>(
    tooffsets,
    outeroffsets,
    outeroffsetslen,
    inneroffsets,
    inneroffsetslen);
}
ERROR awkward_ListOffsetArrayU32_flatten_offsets_64(
  int64_t* tooffsets,
  const uint32_t* outeroffsets,
  int64_t outeroffsetslen,
  const int64_t* inneroffsets,
  int64_t inneroffsetslen) {
  return awkward_ListOffsetArray_flatten_offsets<int64_t, uint32_t>(
    tooffsets,
    outeroffsets,
    outeroffsetslen,
    inneroffsets,
    inneroffsetslen);
}
ERROR awkward_ListOffsetArray64_flatten_offsets_64(
  int64_t* tooffsets,
  const int64_t* outeroffsets,
  int64_t outeroffsetslen,
  const int64_t* inneroffsets,
  int64_t inneroffsetslen) {
  return awkward_ListOffsetArray_flatten_offsets<int64_t, int64_t>(
    tooffsets,
    outeroffsets,
    outeroffsetslen,
    inneroffsets,
    inneroffsetslen);
}
