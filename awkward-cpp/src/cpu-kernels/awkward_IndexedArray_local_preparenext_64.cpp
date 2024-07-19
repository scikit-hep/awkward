// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_local_preparenext_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_IndexedArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* /* starts */,   // used in CUDA kernels
    const int64_t* parents,
    const int64_t parentslength,
    const int64_t* nextparents,
    const int64_t nextlen) {
  int64_t j = 0;
  int64_t parent = 0;
  for (int64_t i = 0;  i < parentslength;  i++) {
    parent = parents[i];
    if (j < nextlen  &&  parent == nextparents[j]) {
      tocarry[i] = j;
      ++j;
    }
    else {
      tocarry[i] = -1;
    }
  }
  return success();
}
