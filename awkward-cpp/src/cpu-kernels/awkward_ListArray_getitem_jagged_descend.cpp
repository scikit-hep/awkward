// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_descend.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_descend(
  T* tooffsets,
  const T* slicestarts,
  const T* slicestops,
  int64_t sliceouterlen,
  const C* fromstarts,
  const C* fromstops) {
  if (sliceouterlen == 0) {
    tooffsets[0] = 0;
  }
  else {
    tooffsets[0] = slicestarts[0];
  }
  for (int64_t i = 0;  i < sliceouterlen;  i++) {
    int64_t slicecount = (int64_t)(slicestops[i] -
                                   slicestarts[i]);
    int64_t count = (int64_t)(fromstops[i] -
                              fromstarts[i]);
    if (slicecount != count) {
      return failure("jagged slice inner length differs from array inner length", i, kSliceNone, FILENAME(__LINE__));
    }
    tooffsets[i + 1] = tooffsets[i] + (T)count;
  }
  return success();
}

#define WRAPPER(SUFFIX, C, T) \
  ERROR awkward_ListArray##SUFFIX(T* tooffsets, const T* slicestarts, const T* slicestops, int64_t sliceouterlen, const C* fromstarts, const C* fromstops) { \
    return awkward_ListArray_getitem_jagged_descend<C, T>(tooffsets, slicestarts, slicestops, sliceouterlen, fromstarts, fromstops); \
  }

WRAPPER(32_getitem_jagged_descend_64, int32_t, int64_t)
WRAPPER(U32_getitem_jagged_descend_64, uint32_t, int64_t)
WRAPPER(64_getitem_jagged_descend_64, int64_t, int64_t)
