// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_index_of_nulls.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_index_of_nulls(
  int64_t* toindex,
  const C* fromindex,
  int64_t lenindex,
  const int64_t* parents,
  const int64_t* starts) {
  int64_t j = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    if (fromindex[i] < 0) {
      int64_t parent = parents[i];
      int64_t start = starts[parent];
      toindex[j++] = i - start;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_index_of_nulls(
  int64_t* toindex,
  const int32_t* fromindex,
  int64_t lenindex,
  const int64_t* parents,
  const int64_t* starts) {
  return awkward_IndexedArray_index_of_nulls<int32_t>(
    toindex,
    fromindex,
    lenindex,
    parents,
    starts);
}
ERROR awkward_IndexedArrayU32_index_of_nulls(
  int64_t* toindex,
  const uint32_t* fromindex,
  int64_t lenindex,
  const int64_t* parents,
  const int64_t* starts) {
  return awkward_IndexedArray_index_of_nulls<uint32_t>(
    toindex,
    fromindex,
    lenindex,
    parents,
    starts);
}
ERROR awkward_IndexedArray64_index_of_nulls(
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t lenindex,
  const int64_t* parents,
  const int64_t* starts) {
  return awkward_IndexedArray_index_of_nulls<int64_t>(
    toindex,
    fromindex,
    lenindex,
    parents,
    starts);
}
