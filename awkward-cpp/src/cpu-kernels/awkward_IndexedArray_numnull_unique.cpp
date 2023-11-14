// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_numnull_unique.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_IndexedArray_numnull_unique_64(
  int64_t* toindex,
  int64_t lenindex) {
  for (int64_t i = 0;  i < lenindex;  i++) {
    toindex[i] = i;
  }
  toindex[lenindex] = -1;
  return success();
}
