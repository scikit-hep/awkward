// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_getitem_carry.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_IndexedArray_getitem_carry(
  C* toindex,
  const C* fromindex,
  const T* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  for (int64_t i = 0;  i < lencarry;  i++) {
    if (fromcarry[i] >= lenindex) {
      return failure("index out of range", i, fromcarry[i], FILENAME(__LINE__));
    }
    toindex[i] = (C)(fromindex[fromcarry[i]]);
  }
  return success();
}
ERROR awkward_IndexedArray32_getitem_carry_64(
  int32_t* toindex,
  const int32_t* fromindex,
  const int64_t* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  return awkward_IndexedArray_getitem_carry<int32_t, int64_t>(
    toindex,
    fromindex,
    fromcarry,
    lenindex,
    lencarry);
}
ERROR awkward_IndexedArrayU32_getitem_carry_64(
  uint32_t* toindex,
  const uint32_t* fromindex,
  const int64_t* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  return awkward_IndexedArray_getitem_carry<uint32_t, int64_t>(
    toindex,
    fromindex,
    fromcarry,
    lenindex,
    lencarry);
}
ERROR awkward_IndexedArray64_getitem_carry_64(
  int64_t* toindex,
  const int64_t* fromindex,
  const int64_t* fromcarry,
  int64_t lenindex,
  int64_t lencarry) {
  return awkward_IndexedArray_getitem_carry<int64_t, int64_t>(
    toindex,
    fromindex,
    fromcarry,
    lenindex,
    lencarry);
}
