// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_at.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_at(
  T* __restrict__ tocarry,
  int64_t at,
  int64_t length,
  int64_t size) {
  const auto regular_at = awkward::kernel::normalize_index(at, size);
  if (!regular_at) {
    return failure("index out of range", kSliceNone, at, FILENAME(__LINE__));
  }
  for (int64_t i = 0;  i < length;  i++) {
    tocarry[i] = i*size + *regular_at;
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_at_64(
  int64_t* __restrict__ tocarry,
  int64_t at,
  int64_t length,
  int64_t size) {
  return awkward_RegularArray_getitem_next_at<int64_t>(
    tocarry,
    at,
    length,
    size);
}
