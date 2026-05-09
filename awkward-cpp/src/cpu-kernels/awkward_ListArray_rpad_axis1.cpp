// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_rpad_axis1.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_ListArray_rpad_axis1(
  T* toindex,
  const C* fromstarts,
  const C* fromstops,
  C* tostarts,
  C* tostops,
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

#define WRAPPER(SUFFIX, T, C) \
  ERROR awkward_ListArray##SUFFIX(T* toindex, const C* fromstarts, const C* fromstops, C* tostarts, C* tostops, int64_t target, int64_t length) { \
    return awkward_ListArray_rpad_axis1<T, C>(toindex, fromstarts, fromstops, tostarts, tostops, target, length); \
  }

WRAPPER(32_rpad_axis1_64, int64_t, int32_t)
WRAPPER(U32_rpad_axis1_64, int64_t, uint32_t)
WRAPPER(64_rpad_axis1_64, int64_t, int64_t)
