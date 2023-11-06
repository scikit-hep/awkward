// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_local_nextparents_64.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListOffsetArray_reduce_local_nextparents_64(
  int64_t* nextparents,
  const C* offsets,
  int64_t length) {
  int64_t initialoffset = (int64_t)(offsets[0]);
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = (int64_t)(offsets[i]) - initialoffset;
         j < offsets[i + 1] - initialoffset;
         j++) {
      nextparents[j] = i;
    }
  }
  return success();
}
ERROR awkward_ListOffsetArray32_reduce_local_nextparents_64(
  int64_t* nextparents,
  const int32_t* offsets,
  int64_t length) {
  return awkward_ListOffsetArray_reduce_local_nextparents_64(
    nextparents,
    offsets,
    length);
}
ERROR awkward_ListOffsetArrayU32_reduce_local_nextparents_64(
  int64_t* nextparents,
  const uint32_t* offsets,
  int64_t length) {
  return awkward_ListOffsetArray_reduce_local_nextparents_64(
    nextparents,
    offsets,
    length);
}
ERROR awkward_ListOffsetArray64_reduce_local_nextparents_64(
  int64_t* nextparents,
  const int64_t* offsets,
  int64_t length) {
  return awkward_ListOffsetArray_reduce_local_nextparents_64(
    nextparents,
    offsets,
    length);
}
