// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ByteMaskedArray_overlay_mask.cpp", line)

#include "awkward/kernels.h"

template <typename M>
ERROR awkward_ByteMaskedArray_overlay_mask(
  M* tomask,
  const M* theirmask,
  const M* mymask,
  int64_t length,
  bool validwhen) {
  for (int64_t i = 0;  i < length;  i++) {
    bool theirs = theirmask[i];
    bool mine = ((mymask[i] != 0) != validwhen);
    tomask[i] = ((theirs | mine) ? 1 : 0);
  }
  return success();
}
ERROR awkward_ByteMaskedArray_overlay_mask8(
  int8_t* tomask,
  const int8_t* theirmask,
  const int8_t* mymask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_overlay_mask<int8_t>(
    tomask,
    theirmask,
    mymask,
    length,
    validwhen);
}
