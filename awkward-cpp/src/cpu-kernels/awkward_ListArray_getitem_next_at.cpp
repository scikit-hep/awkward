// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_at.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_at(
  T* __restrict__ tocarry,
  const C* __restrict__ fromstarts,
  const C* __restrict__ fromstops,
  int64_t lenstarts,
  int64_t at) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[i] - fromstarts[i];
    const auto regular_at = awkward::kernel::normalize_index(at, length);
    if (!regular_at) {
      return failure("index out of range", i, at, FILENAME(__LINE__));
    }
    tocarry[i] = fromstarts[i] + *regular_at;
  }
  return success();
}

#define WRAPPER(FUNC, C, T) \
  ERROR FUNC(T* tocarry, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t at) { \
    return awkward_ListArray_getitem_next_at<C, T>(tocarry, fromstarts, fromstops, lenstarts, at); \
  }

WRAPPER(awkward_ListArray32_getitem_next_at_64, int32_t, int64_t)
WRAPPER(awkward_ListArrayU32_getitem_next_at_64, uint32_t, int64_t)
WRAPPER(awkward_ListArray64_getitem_next_at_64, int64_t, int64_t)
