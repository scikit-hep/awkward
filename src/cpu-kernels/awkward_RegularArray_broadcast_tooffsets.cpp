// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_broadcast_tooffsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_broadcast_tooffsets(
  const T* fromoffsets,
  int64_t offsetslength,
  int64_t size) {
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)(fromoffsets[i + 1] - fromoffsets[i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    if (size != count) {
      return failure("cannot broadcast nested list", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_RegularArray_broadcast_tooffsets_64(
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t size) {
  return awkward_RegularArray_broadcast_tooffsets<int64_t>(
    fromoffsets,
    offsetslength,
    size);
}
