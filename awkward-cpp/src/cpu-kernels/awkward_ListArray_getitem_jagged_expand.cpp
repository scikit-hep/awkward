// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_jagged_expand.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_jagged_expand(
  T* multistarts,
  T* multistops,
  const T* singleoffsets,
  T* tocarry,
  const C* fromstarts,
  const C* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if (stop - start != jaggedsize) {
      return failure("cannot fit jagged slice into nested list", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = 0;  j < jaggedsize;  j++) {
      multistarts[i*jaggedsize + j] = singleoffsets[j];
      multistops[i*jaggedsize + j] = singleoffsets[j + 1];
      tocarry[i*jaggedsize + j] = start + j;
    }
  }
  return success();
}
ERROR awkward_ListArray32_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t* tocarry,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  return awkward_ListArray_getitem_jagged_expand<int32_t, int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length);
}
ERROR awkward_ListArrayU32_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t* tocarry,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  return awkward_ListArray_getitem_jagged_expand<uint32_t, int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length);
}
ERROR awkward_ListArray64_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t* tocarry,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t jaggedsize,
  int64_t length) {
  return awkward_ListArray_getitem_jagged_expand<int64_t, int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length);
}
