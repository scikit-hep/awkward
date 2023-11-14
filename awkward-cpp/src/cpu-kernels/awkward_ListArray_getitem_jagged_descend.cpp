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
ERROR awkward_ListArray32_getitem_jagged_descend_64(
  int64_t* tooffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int32_t* fromstarts,
  const int32_t* fromstops) {
  return awkward_ListArray_getitem_jagged_descend<int32_t, int64_t>(
    tooffsets,
    slicestarts,
    slicestops,
    sliceouterlen,
    fromstarts,
    fromstops);
}
ERROR awkward_ListArrayU32_getitem_jagged_descend_64(
  int64_t* tooffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const uint32_t* fromstarts,
  const uint32_t* fromstops) {
  return awkward_ListArray_getitem_jagged_descend<uint32_t, int64_t>(
    tooffsets,
    slicestarts,
    slicestops,
    sliceouterlen,
    fromstarts,
    fromstops);
}
ERROR awkward_ListArray64_getitem_jagged_descend_64(
  int64_t* tooffsets,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* fromstarts,
  const int64_t* fromstops) {
  return awkward_ListArray_getitem_jagged_descend<int64_t, int64_t>(
    tooffsets,
    slicestarts,
    slicestops,
    sliceouterlen,
    fromstarts,
    fromstops);
}
