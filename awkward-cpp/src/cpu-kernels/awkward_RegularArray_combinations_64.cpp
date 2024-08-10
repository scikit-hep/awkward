// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_combinations_64.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

ERROR awkward_RegularArray_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  int64_t size,
  int64_t length) {
  for (int64_t j = 0;  j < n;  j++) {
    toindex[j] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = size*i;
    int64_t stop = start + size;
    fromindex[0] = start;
    awkward_ListArray_combinations_step_64(
      tocarry,
      toindex,
      fromindex,
      0,
      stop,
      n,
      replacement);
  }
  return success();
}
