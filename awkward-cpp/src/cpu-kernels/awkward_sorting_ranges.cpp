// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
//
// DEPRECATED — parents → offsets converter.
//
// As of the offsets-pipeline migration this kernel is no longer called from
// any of NumpyArray's _unique / _argsort_next / _sort_next paths: callers now
// receive `offsets` directly and skip this conversion. The function is kept
// here for ABI compatibility while downstream consumers migrate. Remove once
// all callers are gone.

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
