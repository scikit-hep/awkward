// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_getitem_nextcarry_outindex.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_IndexedArray_getitem_nextcarry_outindex(
  T* tocarry,
  C* toindex,
  const C* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[i];
    if (j >= lencontent) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else if (j < 0) {
      toindex[i] = -1;
    }
    else {
      tocarry[k] = j;
      toindex[i] = (C)k;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  int32_t* toindex,
  const int32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex<int32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArrayU32_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  uint32_t* toindex,
  const uint32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex<uint32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArray64_getitem_nextcarry_outindex_64(
  int64_t* tocarry,
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_getitem_nextcarry_outindex<int64_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    lenindex,
    lencontent);
}
