// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_subrange_equal_bool.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_NumpyArray_subrange_equal_bool(
  bool* tmpptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {

  bool differ = true;
  int64_t leftlen;
  int64_t rightlen;

  for (int64_t i = 0;  i < length - 1;  i++) {
    leftlen = fromstops[i] - fromstarts[i];
    for (int64_t ii = i + 1; ii < length - 1;  ii++) {
      rightlen = fromstops[ii] - fromstarts[ii];
      if (leftlen == rightlen) {
        differ = false;
        for (int64_t j = 0; j < leftlen; j++) {
          if ((tmpptr[fromstarts[i] + j] != 0) != (tmpptr[fromstarts[ii] + j] != 0)) {
            differ = true;
            break;
          }
        }
      }
    }
  }

  *toequal = !differ;

  return success();
}
