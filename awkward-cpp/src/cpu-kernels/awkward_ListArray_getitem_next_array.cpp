// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_array.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_array(
  T* tocarry,
  T* toadvanced,
  const C* fromstarts,
  const C* fromstops,
  const T* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[i] < fromstarts[i]) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if ((fromstarts[i] != fromstops[i])  &&
        (fromstops[i] > lencontent)) {
      return failure("stops[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
    int64_t length = fromstops[i] - fromstarts[i];
    for (int64_t j = 0;  j < lenarray;  j++) {
      int64_t regular_at = fromarray[j];
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at  &&  regular_at < length)) {
        return failure("index out of range", i, fromarray[j], FILENAME(__LINE__));
      }
      tocarry[i*lenarray + j] = fromstarts[i] + regular_at;
      toadvanced[i*lenarray + j] = j;
    }
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  const int64_t* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array<int32_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent);
}
ERROR awkward_ListArrayU32_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  const int64_t* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array<uint32_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent);
}
ERROR awkward_ListArray64_getitem_next_array_64(
  int64_t* tocarry,
  int64_t* toadvanced,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  const int64_t* fromarray,
  int64_t lenstarts,
  int64_t lenarray,
  int64_t lencontent) {
  return awkward_ListArray_getitem_next_array<int64_t, int64_t>(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent);
}
