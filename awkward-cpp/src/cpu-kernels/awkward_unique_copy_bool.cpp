// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_copy_bool.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_unique_copy_bool(
  const bool* fromptr,
  bool* toptr,
  int64_t length,
  int64_t* tolength) {

  int64_t j = 0;
  toptr[0] = fromptr[0];
  for (int64_t i = 1;  i < length;  i++) {
    if ((toptr[j] != 0) != (fromptr[i] != 0)) {
      j++;
      toptr[j] = fromptr[i];
    }
  }
  *tolength = j + 1;
  return success();
}
