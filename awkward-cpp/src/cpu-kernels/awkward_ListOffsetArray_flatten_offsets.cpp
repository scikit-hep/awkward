// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_flatten_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListOffsetArray_flatten_offsets(
  T* __restrict__ tooffsets,
  const C* __restrict__ outeroffsets,
  int64_t outeroffsetslen,
  const T* __restrict__ inneroffsets) {
  for (int64_t i = 0;  i < outeroffsetslen;  i++) {
    tooffsets[i] =
      inneroffsets[outeroffsets[i]];
  }
  return success();
}

#define WRAPPER(FUNC, T, C) \
  ERROR FUNC(T* tooffsets, const C* outeroffsets, int64_t outeroffsetslen, const T* inneroffsets) { \
    return awkward_ListOffsetArray_flatten_offsets<T, C>(tooffsets, outeroffsets, outeroffsetslen, inneroffsets); \
  }

WRAPPER(awkward_ListOffsetArray32_flatten_offsets_64, int64_t, int32_t)
WRAPPER(awkward_ListOffsetArrayU32_flatten_offsets_64, int64_t, uint32_t)
WRAPPER(awkward_ListOffsetArray64_flatten_offsets_64, int64_t, int64_t)
