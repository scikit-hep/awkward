// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_bool_64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique_bool_64(
  T* toptr,
  int64_t length,
  int64_t* tolength) {

  int64_t j = 0;
  for (int64_t i = 1;  i < length;  i++) {
    if ((toptr[j] != 0) != (toptr[i] != 0)) {
      j++;
      toptr[j] = toptr[i];
    }
  }
  *tolength = j + 1;
  return success();
}
