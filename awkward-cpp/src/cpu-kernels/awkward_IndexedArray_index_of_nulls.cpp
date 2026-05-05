// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_index_of_nulls.cpp", line)

#include "awkward/kernels.h"

template <typename C>
ERROR awkward_IndexedArray_index_of_nulls(
  int64_t* toindex,
  const C* fromindex,
  const int64_t* offsets,
  int64_t outlength,
  const int64_t* starts) {
  int64_t j = 0;
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t start = starts[bin];
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      if (fromindex[i] < 0) {
        toindex[j++] = i - start;
      }
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_index_of_nulls(
  int64_t* toindex,
  const int32_t* fromindex,
  const int64_t* offsets,
  int64_t outlength,
  const int64_t* starts) {
  return awkward_IndexedArray_index_of_nulls<int32_t>(
    toindex,
    fromindex,
    offsets,
    outlength,
    starts);
}
ERROR awkward_IndexedArrayU32_index_of_nulls(
  int64_t* toindex,
  const uint32_t* fromindex,
  const int64_t* offsets,
  int64_t outlength,
  const int64_t* starts) {
  return awkward_IndexedArray_index_of_nulls<uint32_t>(
    toindex,
    fromindex,
    offsets,
    outlength,
    starts);
}
ERROR awkward_IndexedArray64_index_of_nulls(
  int64_t* toindex,
  const int64_t* fromindex,
  const int64_t* offsets,
  int64_t outlength,
  const int64_t* starts) {
  return awkward_IndexedArray_index_of_nulls<int64_t>(
    toindex,
    fromindex,
    offsets,
    outlength,
    starts);
}
