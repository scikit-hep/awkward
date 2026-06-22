// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_expand.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_expand(
  T* __restrict__ multistarts,
  T* __restrict__ multistops,
  const T* __restrict__ singleoffsets,
  T* __restrict__ tocarry,
  const C* __restrict__ fromstarts,
  const C* __restrict__ fromstops,
  int64_t jaggedsize,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if (stop - start != jaggedsize) {
      return failure("cannot fit jagged slice into nested list", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = 0;  j < jaggedsize;  j++) {
      multistarts[i*jaggedsize + j] = singleoffsets[j];
      multistops[i*jaggedsize + j] = singleoffsets[j + 1];
      tocarry[i*jaggedsize + j] = start + j;
    }
  }
  return success();
}

#define WRAPPER(FUNC, C, T) \
  ERROR FUNC(T* multistarts, T* multistops, const T* singleoffsets, T* tocarry, const C* fromstarts, const C* fromstops, int64_t jaggedsize, int64_t length) { \
    return awkward_ListArray_getitem_jagged_expand<C, T>(multistarts, multistops, singleoffsets, tocarry, fromstarts, fromstops, jaggedsize, length); \
  }

WRAPPER(awkward_ListArray32_getitem_jagged_expand_64, int32_t, int64_t)
WRAPPER(awkward_ListArrayU32_getitem_jagged_expand_64, uint32_t, int64_t)
WRAPPER(awkward_ListArray64_getitem_jagged_expand_64, int64_t, int64_t)
