// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_getitem_boolmask.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_getitem_boolmask_64(
  int64_t* __restrict__ tooffsets,
  int64_t* __restrict__ tocarry,
  const int8_t* __restrict__ mask,
  const int64_t* __restrict__ fromoffsets,
  int64_t length,
  int64_t carrylength) {
  int64_t k = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = fromoffsets[i];
    int64_t stop = fromoffsets[i + 1];
    for (int64_t j = start;  j < stop;  j++) {
      // the position is written unconditionally and only the cursor advances,
      // because a mask-dependent branch mispredicts on unsorted data; the
      // bounds test is true for every element but the last unselected one
      if (k < carrylength) {
        tocarry[k] = j - start;
      }
      k += (mask[j] != 0);
    }
    tooffsets[i + 1] = k;
  }
  return success();
}
