// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_numnull_parents.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_numnull_parents(
  int64_t* numnull,
  int64_t* tolength,
  const C* fromindex,
  int64_t lenindex) {
  *tolength = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    if (fromindex[i] < 0) {
      numnull[i] = 1;
      *tolength = *tolength + 1;
    }
    else {
      numnull[i] = 0;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_numnull_parents(
  int64_t* numnull,
  int64_t* tolength,
  const int32_t* fromindex,
  int64_t lenindex) {
  return awkward_IndexedArray_numnull_parents<int32_t>(
    numnull,
    tolength,
    fromindex,
    lenindex);
}
ERROR awkward_IndexedArrayU32_numnull_parents(
  int64_t* numnull,
  int64_t* tolength,
  const uint32_t* fromindex,
  int64_t lenindex) {
  return awkward_IndexedArray_numnull_parents<uint32_t>(
    numnull,
    tolength,
    fromindex,
    lenindex);
}
ERROR awkward_IndexedArray64_numnull_parents(
  int64_t* numnull,
  int64_t* tolength,
  const int64_t* fromindex,
  int64_t lenindex) {
  return awkward_IndexedArray_numnull_parents<int64_t>(
    numnull,
    tolength,
    fromindex,
    lenindex);
}
