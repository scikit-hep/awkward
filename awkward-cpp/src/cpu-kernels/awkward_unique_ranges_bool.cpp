// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_ranges_bool.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_unique_ranges_bool(
  bool* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
  int64_t m = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    tooffsets[i] = m;
    toptr[m++] = toptr[fromoffsets[i]];
    for (int64_t k = fromoffsets[i]; k < fromoffsets[i + 1]; k++) {
      if ((toptr[m - 1] != 0) != (toptr[k] != 0)) {
        toptr[m++] = toptr[k];
      }
    }
  }
  tooffsets[offsetslength - 1] = m;

  return success();
}
