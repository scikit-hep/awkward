// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmax.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_reduce_argmax_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t k = 0;  k < outlength;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (toptr[parent] == -1  ||  (fromptr[i] != 0) > (fromptr[toptr[parent]] != 0)) {
      toptr[parent] = i;
    }
  }
  return success();
}
