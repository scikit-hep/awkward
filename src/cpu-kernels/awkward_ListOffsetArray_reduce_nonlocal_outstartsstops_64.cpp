// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
  int64_t* outstarts,
  int64_t* outstops,
  const int64_t* distincts,
  int64_t lendistincts,
  const int64_t* gaps,
  int64_t outlength) {
  int64_t j = 0;
  int64_t k = 0;
  int64_t maxdistinct = -1;
  for (int64_t i = 0;  i < lendistincts;  i++) {
    if (maxdistinct < distincts[i]) {
      maxdistinct = distincts[i];
      for (int64_t gappy = 0;  gappy < gaps[j];  gappy++) {
        outstarts[k] = i;
        outstops[k] = i;
        k++;
      }
      j++;
    }
    if (distincts[i] != -1) {
      outstops[k - 1] = i + 1;
    }
  }
  for (;  k < outlength;  k++) {
    outstarts[k] = lendistincts + 1;
    outstops[k] = lendistincts + 1;
  }
  return success();
}
