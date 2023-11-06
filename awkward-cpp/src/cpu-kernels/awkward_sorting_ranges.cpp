// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sorting_ranges.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_sorting_ranges(
  int64_t* toindex,
  int64_t tolength,
  const int64_t* parents,
  int64_t parentslength) {
  int64_t j = 0;
  int64_t k = 0;
  toindex[0] = k;
  k++; j++;
  for (int64_t i = 1;  i < parentslength;  i++) {
    if (parents[i - 1] != parents[i]) {
      toindex[j] = k;
      j++;
    }
    k++;
  }
  toindex[tolength - 1] = parentslength;
  return success();
}
