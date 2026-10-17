// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_rpad_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListArray_rpad_axis1(
  T* __restrict__ toindex,
  const C* __restrict__ fromstarts,
  const C* __restrict__ fromstops,
  C* __restrict__ tostarts,
  C* __restrict__ tostops,
  int64_t target,
  int64_t length) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    int64_t rangeval = fromstops[i] - fromstarts[i];
    for (int64_t j = 0; j < rangeval; j++) {
     toindex[offset + j] = fromstarts[i] + j;
    }
    for (int64_t j = rangeval; j < target; j++) {
     toindex[offset + j] = -1;
    }
    offset = (target > rangeval) ? tostarts[i] + target : tostarts[i] + rangeval;
    tostops[i] = offset;
  }
  return success();
}

#define WRAPPER(FUNC, T, C) \
  ERROR FUNC(T* toindex, const C* fromstarts, const C* fromstops, C* tostarts, C* tostops, int64_t target, int64_t length) { \
    return awkward_ListArray_rpad_axis1<T, C>(toindex, fromstarts, fromstops, tostarts, tostops, target, length); \
  }

WRAPPER(awkward_ListArray32_rpad_axis1_64, int64_t, int32_t)
WRAPPER(awkward_ListArrayU32_rpad_axis1_64, int64_t, uint32_t)
WRAPPER(awkward_ListArray64_rpad_axis1_64, int64_t, int64_t)
