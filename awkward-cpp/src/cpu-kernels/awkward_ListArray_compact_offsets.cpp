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

#define WRAPPER(SUFFIX, C, T) \
  ERROR awkward_ListArray##SUFFIX(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t length) { \
    return awkward_ListArray_compact_offsets<C, T>(tooffsets, fromstarts, fromstops, length); \
  }

WRAPPER(32_compact_offsets_64, int32_t, int64_t)
WRAPPER(U32_compact_offsets_64, uint32_t, int64_t)
WRAPPER(64_compact_offsets_64, int64_t, int64_t)
