// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_compact_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_compact_offsets(
  T* tooffsets,
  const C* fromstarts,
  const C* fromstops,
  int64_t length) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
  }
  return success();
}
ERROR awkward_ListArray32_compact_offsets_64(
  int64_t* tooffsets,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_compact_offsets<int32_t, int64_t>(
    tooffsets,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArrayU32_compact_offsets_64(
  int64_t* tooffsets,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_compact_offsets<uint32_t, int64_t>(
    tooffsets,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArray64_compact_offsets_64(
  int64_t* tooffsets,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length) {
  return awkward_ListArray_compact_offsets<int64_t, int64_t>(
    tooffsets,
    fromstarts,
    fromstops,
    length);
}
