// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_carrylen.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListArray_getitem_jagged_carrylen(
  int64_t* carrylen,
  const T* slicestarts,
  const T* slicestops,
  int64_t sliceouterlen) {
  *carrylen = 0;
  for (int64_t i = 0;  i < sliceouterlen;  i++) {
    *carrylen = *carrylen + (int64_t)(slicestops[i] - slicestarts[i]);
  }
  return success();
}
ERROR awkward_ListArray_getitem_jagged_carrylen_64(
  int64_t* carrylen,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen) {
  return awkward_ListArray_getitem_jagged_carrylen<int64_t>(
    carrylen,
    slicestarts,
    slicestops,
    sliceouterlen);
}
