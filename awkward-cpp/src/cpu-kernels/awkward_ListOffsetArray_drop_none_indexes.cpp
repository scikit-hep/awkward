// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_drop_none_indexes.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListOffsetArray_drop_none_indexes(
  T* __restrict__ tooffsets,
  const T* __restrict__ noneindexes,
  const T* __restrict__ fromoffsets,
  int64_t length_offsets,
  int64_t length_indexes) {
  T nr_of_nones = 0;
  int64_t offset1 = 0;
  int64_t offset2 = 0;

  for (int64_t i = 0; i < length_offsets; i++) {
    if ((int64_t)fromoffsets[i] > length_indexes) {
      return failure("offsets[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
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

#define WRAPPER(FUNC, T) \
  ERROR FUNC(T* tooffsets, const T* noneindexes, const T* fromoffsets, int64_t length_offsets, int64_t length_indexes) { \
    return awkward_ListOffsetArray_drop_none_indexes<T>(tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes); \
  }

WRAPPER(awkward_ListOffsetArray_drop_none_indexes_64, int64_t)
WRAPPER(awkward_ListOffsetArray_drop_none_indexes_32, int32_t)
