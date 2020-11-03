// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_boolean_numtrue.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_getitem_boolean_numtrue(
  int64_t* numtrue,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
  *numtrue = 0;
  for (int64_t i = 0;  i < length;  i += stride) {
    *numtrue = *numtrue + (fromptr[i] != 0);
  }
  return success();
}
