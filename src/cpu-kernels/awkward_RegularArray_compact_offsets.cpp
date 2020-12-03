// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_compact_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_compact_offsets(
  T* tooffsets,
  int64_t length,
  int64_t size) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[i + 1] = (i + 1)*size;
  }
  return success();
}
ERROR awkward_RegularArray_compact_offsets64(
  int64_t* tooffsets,
  int64_t length,
  int64_t size) {
  return awkward_RegularArray_compact_offsets<int64_t>(
    tooffsets,
    length,
    size);
}
