// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_drop_none_indexes.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListOffsetArray_drop_none_indexes(
  T* tooffsets,
  const T* noneindexes,
  const T* fromoffsets,
  int64_t length_offsets,
  int64_t length_indexes) {
  if (length_offsets > 0 && fromoffsets[length_offsets - 1] > length_indexes) {
    return failure("offsets[i] > len(content)", kSliceNone, kSliceNone, FILENAME(__LINE__));
  }
  T nr_of_nones = 0;
  int64_t offset1 = 0;
  int64_t offset2 = 0;

  for (int64_t i = 0; i < length_offsets; i++) {
    offset2 = fromoffsets[i];
    for (int64_t j = offset1; j < offset2; j++) {
      nr_of_nones += (noneindexes[j] < 0);
    }
    tooffsets[i] = fromoffsets[i] - nr_of_nones;
    offset1 = offset2;
  }

  return success();
}
ERROR awkward_ListOffsetArray_drop_none_indexes_64(
  int64_t* tooffsets,
  const int64_t* noneindexes,
  const int64_t* fromoffsets,
  int64_t length_offsets,
  int64_t length_indexes) {
  return awkward_ListOffsetArray_drop_none_indexes<int64_t>(
    tooffsets,
    noneindexes,
    fromoffsets,
    length_offsets,
    length_indexes);
}
ERROR awkward_ListOffsetArray_drop_none_indexes_32(
  int32_t* tooffsets,
  const int32_t* noneindexes,
  const int32_t* fromoffsets,
  int64_t length_offsets,
  int64_t length_indexes) {
  return awkward_ListOffsetArray_drop_none_indexes<int32_t>(
    tooffsets,
    noneindexes,
    fromoffsets,
    length_offsets,
    length_indexes);
}
