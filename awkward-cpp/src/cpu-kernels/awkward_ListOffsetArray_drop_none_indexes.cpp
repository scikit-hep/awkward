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
  T nr_of_nones = 0;
  int64_t offset1 = 0;
  int64_t offset2 = 0;

  if (length_offsets > 0 && fromoffsets[length_offsets - 1] > length_indexes) {
    return failure("offsets[i] > len(content)", length_offsets - 1, kSliceNone, FILENAME(__LINE__));
  }
  for (int64_t i = 0; i < length_offsets; i++) {
    offset2 = fromoffsets[i];
    for (int64_t j = offset1; j < offset2; j++) {
        if (noneindexes[j] < 0) {
            nr_of_nones++;
        }
    }
    tooffsets[i] = fromoffsets[i] - nr_of_nones;
    offset1 = offset2;
  }

  return success();
}

#define WRAPPER(SUFFIX, T) \
  ERROR awkward_ListOffsetArray_drop_none_indexes_##SUFFIX(T* tooffsets, const T* noneindexes, const T* fromoffsets, int64_t length_offsets, int64_t length_indexes) { \
    return awkward_ListOffsetArray_drop_none_indexes<T>(tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes); \
  }

WRAPPER(64, int64_t)
WRAPPER(32, int32_t)
