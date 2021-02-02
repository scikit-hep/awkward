// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_SliceVarNewAxis_to_SliceJagged64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_SliceVarNewAxis_to_SliceJagged64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = fromoffsets[i];
    int64_t stop = fromoffsets[i + 1];
    for (int64_t j = start;  j < stop;  j++) {
      tocarry[j] = i;
    }
  }
  return success();
}
