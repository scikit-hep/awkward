// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_validity.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_ListArray_validity(
  const C* __restrict__ starts,
  const C* __restrict__ stops,
  int64_t length,
  int64_t lencontent) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = starts[i];
    C stop = stops[i];
    if (start != stop) {
      if (start > stop) {
        return failure("start[i] > stop[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (start < 0) {
        return failure("start[i] < 0", i, kSliceNone, FILENAME(__LINE__));
      }
      if (stop > lencontent) {
        return failure("stop[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
      }
    }
  }
  return success();
}

#define WRAPPER(FUNC, C) \
  ERROR FUNC(const C* starts, const C* stops, int64_t length, int64_t lencontent) { \
    return awkward_ListArray_validity<C>(starts, stops, length, lencontent); \
  }

WRAPPER(awkward_ListArray32_validity, int32_t)
WRAPPER(awkward_ListArrayU32_validity, uint32_t)
WRAPPER(awkward_ListArray64_validity, int64_t)
