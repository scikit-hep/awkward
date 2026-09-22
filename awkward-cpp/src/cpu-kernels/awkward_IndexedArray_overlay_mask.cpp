// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_overlay_mask.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename M, typename TO>
ERROR awkward_IndexedArray_overlay_mask(
  TO* __restrict__ toindex,
  const M* __restrict__ mask,
  const C* __restrict__ fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    M m = mask[i];
    toindex[i] = (m ? -1 : fromindex[i]);
  }
  return success();
}

#define WRAPPER(FUNC, C, M, TO) \
  ERROR FUNC(TO* toindex, const M* mask, const C* fromindex, int64_t length) { \
    return awkward_IndexedArray_overlay_mask<C, M, TO>(toindex, mask, fromindex, length); \
  }

WRAPPER(awkward_IndexedArray32_overlay_mask8_to64, int32_t, int8_t, int64_t)
WRAPPER(awkward_IndexedArrayU32_overlay_mask8_to64, uint32_t, int8_t, int64_t)
WRAPPER(awkward_IndexedArray64_overlay_mask8_to64, int64_t, int8_t, int64_t)
