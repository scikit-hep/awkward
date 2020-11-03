// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_getitem_adjust_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListOffsetArray_getitem_adjust_offsets(
  T* tooffsets,
  T* tononzero,
  const T* fromoffsets,
  int64_t length,
  const T* nonzero,
  int64_t nonzerolength) {
  int64_t j = 0;
  tooffsets[0] = fromoffsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = fromoffsets[i];
    T slicestop = fromoffsets[i + 1];
    int64_t count = 0;
    while (j < nonzerolength  &&  nonzero[j] < slicestop) {
      tononzero[j] = nonzero[j] - slicestart;
      j++;
      count++;
    }
    tooffsets[i + 1] = tooffsets[i] + count;
  }
  return success();
}
ERROR awkward_ListOffsetArray_getitem_adjust_offsets_64(
  int64_t* tooffsets,
  int64_t* tononzero,
  const int64_t* fromoffsets,
  int64_t length,
  const int64_t* nonzero,
  int64_t nonzerolength) {
  return awkward_ListOffsetArray_getitem_adjust_offsets<int64_t>(
    tooffsets,
    tononzero,
    fromoffsets,
    length,
    nonzero,
    nonzerolength);
}
