// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_broadcast_tooffsets.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_broadcast_tooffsets(
  T* tocarry,
  const T* fromoffsets,
  int64_t offsetslength,
  const C* fromstarts,
  const C* fromstops,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t start = (int64_t)fromstarts[i];
    int64_t stop = (int64_t)fromstops[i];
    if (start != stop  &&  stop > lencontent) {
      return failure("stops[i] > len(content)", i, stop, FILENAME(__LINE__));
    }
    int64_t count = (int64_t)(fromoffsets[i + 1] - fromoffsets[i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    if (stop - start != count) {
      return failure("cannot broadcast nested list", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = start;  j < stop;  j++) {
      tocarry[k] = (T)j;
      k++;
    }
  }
  return success();
}
ERROR awkward_ListArray32_broadcast_tooffsets_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lencontent) {
  return awkward_ListArray_broadcast_tooffsets<int32_t, int64_t>(
    tocarry,
    fromoffsets,
    offsetslength,
    fromstarts,
    fromstops,
    lencontent);
}
ERROR awkward_ListArrayU32_broadcast_tooffsets_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lencontent) {
  return awkward_ListArray_broadcast_tooffsets<uint32_t, int64_t>(
    tocarry,
    fromoffsets,
    offsetslength,
    fromstarts,
    fromstops,
    lencontent);
}
ERROR awkward_ListArray64_broadcast_tooffsets_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lencontent) {
  return awkward_ListArray_broadcast_tooffsets<int64_t, int64_t>(
    tocarry,
    fromoffsets,
    offsetslength,
    fromstarts,
    fromstops,
    lencontent);
}
