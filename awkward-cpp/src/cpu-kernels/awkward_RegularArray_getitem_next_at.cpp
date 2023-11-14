// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_at.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_at(
  T* tocarry,
  int64_t at,
  int64_t length,
  int64_t size) {
  int64_t regular_at = at;
  if (regular_at < 0) {
    regular_at += size;
  }
  if (!(0 <= regular_at  &&  regular_at < size)) {
    return failure("index out of range", kSliceNone, at, FILENAME(__LINE__));
  }
  for (int64_t i = 0;  i < length;  i++) {
    tocarry[i] = i*size + regular_at;
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_at_64(
  int64_t* tocarry,
  int64_t at,
  int64_t length,
  int64_t size) {
  return awkward_RegularArray_getitem_next_at<int64_t>(
    tocarry,
    at,
    length,
    size);
}
