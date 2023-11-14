// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_apply.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_apply(
  T* tooffsets,
  T* tocarry,
  const T* slicestarts,
  const T* slicestops,
  int64_t sliceouterlen,
  const T* sliceindex,
  int64_t sliceinnerlen,
  const C* fromstarts,
  const C* fromstops,
  int64_t contentlen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < sliceouterlen;  i++) {
    T slicestart = slicestarts[i];
    T slicestop = slicestops[i];
    tooffsets[i] = (T)k;
    if (slicestart != slicestop) {
      if (slicestop < slicestart) {
        return failure("jagged slice's stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (slicestop > sliceinnerlen) {
        return failure("jagged slice's offsets extend beyond its content", i, slicestop, FILENAME(__LINE__));
      }
      int64_t start = (int64_t)fromstarts[i];
      int64_t stop = (int64_t)fromstops[i];
      if (stop < start) {
        return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (start != stop  &&  stop > contentlen) {
        return failure("stops[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
      }
      int64_t count = stop - start;
      for (int64_t j = slicestart;  j < slicestop;  j++) {
        int64_t index = (int64_t) sliceindex[j];
        if (index < -count || index > count) {
          return failure("index out of range", i, index, FILENAME(__LINE__));
        }
        if (index < 0) {
          index += count;
        }
        tocarry[k] = start + index;
        k++;
      }
    }
  }
  tooffsets[sliceouterlen] = (T)k;
  return success();
}
ERROR awkward_ListArray32_getitem_jagged_apply_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* sliceindex,
  int64_t sliceinnerlen,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t contentlen) {
  return awkward_ListArray_getitem_jagged_apply<int32_t, int64_t>(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen);
}
ERROR awkward_ListArrayU32_getitem_jagged_apply_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* sliceindex,
  int64_t sliceinnerlen,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t contentlen) {
  return awkward_ListArray_getitem_jagged_apply<uint32_t, int64_t>(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen);
}
ERROR awkward_ListArray64_getitem_jagged_apply_64(
  int64_t* tooffsets,
  int64_t* tocarry,
  const int64_t* slicestarts,
  const int64_t* slicestops,
  int64_t sliceouterlen,
  const int64_t* sliceindex,
  int64_t sliceinnerlen,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t contentlen) {
  return awkward_ListArray_getitem_jagged_apply<int64_t, int64_t>(
    tooffsets,
    tocarry,
    slicestarts,
    slicestops,
    sliceouterlen,
    sliceindex,
    sliceinnerlen,
    fromstarts,
    fromstops,
    contentlen);
}
