// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_at.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_at(
  T* tocarry,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts,
  int64_t at) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure("index out of range", i, at, FILENAME(__LINE__));
    }
    tocarry[i] = fromstarts[i] + regular_at;
  }
  return success();
}
ERROR awkward_ListArray32_getitem_next_at_64(
  int64_t* tocarry,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts,
  int64_t at) {
  return awkward_ListArray_getitem_next_at<int32_t, int64_t>(
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    at);
}
ERROR awkward_ListArrayU32_getitem_next_at_64(
  int64_t* tocarry,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts,
  int64_t at) {
  return awkward_ListArray_getitem_next_at<uint32_t, int64_t>(
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    at);
}
ERROR awkward_ListArray64_getitem_next_at_64(
  int64_t* tocarry,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts,
  int64_t at) {
  return awkward_ListArray_getitem_next_at<int64_t, int64_t>(
    tocarry,
    fromstarts,
    fromstops,
    lenstarts,
    at);
}
