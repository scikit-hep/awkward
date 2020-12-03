// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_getitem_adjust_offsets_index.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListOffsetArray_getitem_adjust_offsets_index(
  T* tooffsets,
  T* tononzero,
  const T* fromoffsets,
  int64_t length,
  const T* index,
  int64_t indexlength,
  const T* nonzero,
  int64_t nonzerolength,
  const int8_t* originalmask,
  int64_t masklength) {
  int64_t k = 0;
  tooffsets[0] = fromoffsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = fromoffsets[i];
    T slicestop = fromoffsets[i + 1];
    int64_t numnull = 0;
    for (int64_t j = slicestart;  j < slicestop;  j++) {
      numnull += (originalmask[j] != 0 ? 1 : 0);
    }
    int64_t nullcount = 0;
    int64_t count = 0;
    while (k < indexlength  &&
           ((index[k] < 0  && nullcount < numnull)  ||
            (index[k] >= 0  &&
             index[k] < nonzerolength  &&
             nonzero[index[k]] < slicestop))) {
      if (index[k] < 0) {
        nullcount++;
      }
      else {
        int64_t j = index[k];
        tononzero[j] = nonzero[j] - slicestart;
      }
      k++;
      count++;
    }
    tooffsets[i + 1] = tooffsets[i] + count;
  }
  return success();
}
ERROR awkward_ListOffsetArray_getitem_adjust_offsets_index_64(
  int64_t* tooffsets,
  int64_t* tononzero,
  const int64_t* fromoffsets,
  int64_t length,
  const int64_t* index,
  int64_t indexlength,
  const int64_t* nonzero,
  int64_t nonzerolength,
  const int8_t* originalmask,
  int64_t masklength) {
  return awkward_ListOffsetArray_getitem_adjust_offsets_index<int64_t>(
    tooffsets,
    tononzero,
    fromoffsets,
    length,
    index,
    indexlength,
    nonzero,
    nonzerolength,
    originalmask,
    masklength);
}
