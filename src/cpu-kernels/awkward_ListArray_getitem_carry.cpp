// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_carry.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_carry(
  C* tostarts,
  C* tostops,
  const C* fromstarts,
  const C* fromstops,
  const T* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenstarts) {
      return failure("index out of range", i, fromcarry[i], FILENAME(__LINE__));
    }
    tostarts[i] = (C)(fromstarts[fromcarry[i]]);
    tostops[i] = (C)(fromstops[fromcarry[i]]);
  }
  return success();
}
ERROR awkward_ListArray32_getitem_carry_64(
  int32_t* tostarts,
  int32_t* tostops,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  const int64_t* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  return awkward_ListArray_getitem_carry<int32_t, int64_t>(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    lenstarts,
    lencarry);
}
ERROR awkward_ListArrayU32_getitem_carry_64(
  uint32_t* tostarts,
  uint32_t* tostops,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  const int64_t* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  return awkward_ListArray_getitem_carry<uint32_t, int64_t>(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    lenstarts,
    lencarry);
}
ERROR awkward_ListArray64_getitem_carry_64(
  int64_t* tostarts,
  int64_t* tostops,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  const int64_t* fromcarry,
  int64_t lenstarts,
  int64_t lencarry) {
  return awkward_ListArray_getitem_carry<int64_t, int64_t>(
    tostarts,
    tostops,
    fromstarts,
    fromstops,
    fromcarry,
    lenstarts,
    lencarry);
}
