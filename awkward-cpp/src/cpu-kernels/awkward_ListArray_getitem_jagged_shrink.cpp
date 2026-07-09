// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_shrink.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListArray_getitem_jagged_shrink(
  T* __restrict__ tocarry,
  T* __restrict__ tosmalloffsets,
  T* __restrict__ tolargeoffsets,
  const T* __restrict__ slicestarts,
  const T* __restrict__ slicestops,
  int64_t length,
  const T* __restrict__ missing) {
  int64_t k = 0;
  if (length == 0) {
    tosmalloffsets[0] = 0;
    tolargeoffsets[0] = 0;
  }
  else {
    tosmalloffsets[0] = slicestarts[0];
    tolargeoffsets[0] = slicestarts[0];
  }
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = slicestarts[i];
    T slicestop = slicestops[i];
    if (slicestart != slicestop) {
      T smallcount = 0;
      for (int64_t j = slicestart;  j < slicestop;  j++) {
        if (missing[j] >= 0) {
          tocarry[k] = j;
          k++;
          smallcount++;
        }
      }
      tosmalloffsets[i + 1] = tosmalloffsets[i] + smallcount;
    }
    else {
      tosmalloffsets[i + 1] = tosmalloffsets[i];
    }
    tolargeoffsets[i + 1] = tolargeoffsets[i] + (slicestop - slicestart);
  }
  return success();
}
ERROR awkward_ListArray_getitem_jagged_shrink_64(
  int64_t* __restrict__ tocarry,
  int64_t* __restrict__ tosmalloffsets,
  int64_t* __restrict__ tolargeoffsets,
  const int64_t* __restrict__ slicestarts,
  const int64_t* __restrict__ slicestops,
  int64_t length,
  const int64_t* __restrict__ missing) {
  return awkward_ListArray_getitem_jagged_shrink<int64_t>(
    tocarry,
    tosmalloffsets,
    tolargeoffsets,
    slicestarts,
    slicestops,
    length,
    missing);
}
