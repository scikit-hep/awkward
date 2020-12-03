// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_copy.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_copy(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len) {
  memcpy(toptr, fromptr, (size_t)len);
  return success();
}
