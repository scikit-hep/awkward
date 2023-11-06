// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_reduce_next_64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_IndexedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const T* index,
  const int64_t* parents,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (index[i] >= 0) {
      nextcarry[k] = index[i];
      nextparents[k] = parents[i];
      outindex[i] = k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int32_t* index,
  int64_t* parents,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_64<int32_t>(
    nextcarry,
    nextparents,
    outindex,
    index,
    parents,
    length);
}
ERROR awkward_IndexedArrayU32_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const uint32_t* index,
  int64_t* parents,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_64<uint32_t>(
    nextcarry,
    nextparents,
    outindex,
    index,
    parents,
    length);
}
ERROR awkward_IndexedArray64_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int64_t* index,
  int64_t* parents,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_64<int64_t>(
    nextcarry,
    nextparents,
    outindex,
    index,
    parents,
    length);
}
