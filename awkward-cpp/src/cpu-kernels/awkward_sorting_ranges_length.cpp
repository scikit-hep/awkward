// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sorting_ranges_length.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_sorting_ranges_length(
  int64_t* tolength,
  const int64_t* parents,
  int64_t parentslength) {
  int64_t length = 2;
  for (int64_t i = 1;  i < parentslength;  i++) {
    if (parents[i - 1] != parents[i]) {
      length++;
    }
  }
  *tolength = length;
  return success();
}
