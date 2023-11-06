// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_numvalid.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_ListArray_getitem_jagged_numvalid(
  int64_t* numvalid,
  const T* slicestarts,
  const T* slicestops,
  int64_t length,
  const T* missing,
  int64_t missinglength) {
  *numvalid = 0;
  for (int64_t i = 0;  i < length;  i++) {
    T slicestart = slicestarts[i];
    T slicestop = slicestops[i];
    if (slicestart != slicestop) {
      if (slicestop < slicestart) {
        return failure("jagged slice's stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (slicestop > missinglength) {
        return failure("jagged slice's offsets extend beyond its content", i, slicestop, FILENAME(__LINE__));
      }
      for (int64_t j = slicestart;  j < slicestop;  j++) {
        *numvalid = *numvalid + (missing[j] >= 0 ? 1 : 0);
      }
    }
  }
  return success();
}
ERROR awkward_ListArray_getitem_jagged_numvalid_64(
  int64_t* numvalid,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t length,
  const int64_t* missing,
  int64_t missinglength) {
  return awkward_ListArray_getitem_jagged_numvalid<int64_t>(
    numvalid,
    slicestarts,
    slicestops,
    length,
    missing,
    missinglength);
}
