// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64.cpp", line)
#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
  int64_t* outstarts,
  int64_t* outstops,
  const int64_t* distincts,
  int64_t lendistincts,
  int64_t outlength) {
  int64_t maxcount = (outlength == 0) ? lendistincts : lendistincts / outlength;

  int64_t k = 0;
  int64_t i_next = 0;
  for (int64_t i = 0;  i < lendistincts;  i++) {
    // Did we hit the start of the next sublist in `distincts`?
    if (i == i_next) {
      i_next += maxcount;

      // Add a new sublist, which is empty by default
      outstarts[k] = i;
      outstops[k] = i;
      k++;
    }

    if (distincts[i] != -1) {
      outstops[k-1] = i+1;
    }
  }
  // assert (k == outlength);
  return success();
}
