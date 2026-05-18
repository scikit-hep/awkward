// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_overlay_mask.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename M, typename TO>
ERROR awkward_IndexedArray_overlay_mask(
  TO* toindex,
  const M* mask,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    M m = mask[i];
    toindex[i] = (m ? -1 : fromindex[i]);
  }
  return success();
}

#define WRAPPER(SUFFIX, C, M, TO) \
  ERROR awkward_IndexedArray##SUFFIX(TO* toindex, const M* mask, const C* fromindex, int64_t length) { \
    return awkward_IndexedArray_overlay_mask<C, M, TO>(toindex, mask, fromindex, length); \
  }

WRAPPER(32_overlay_mask8_to64, int32_t, int8_t, int64_t)
WRAPPER(U32_overlay_mask8_to64, uint32_t, int8_t, int64_t)
WRAPPER(64_overlay_mask8_to64, int64_t, int8_t, int64_t)
