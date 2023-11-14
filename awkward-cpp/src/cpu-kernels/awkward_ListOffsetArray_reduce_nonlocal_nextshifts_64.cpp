// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64(
  int64_t* nummissing,
  int64_t* missing,
  int64_t* nextshifts,
  const int64_t* offsets,
  int64_t length,
  const int64_t* starts,
  const int64_t* parents,
  int64_t maxcount,
  int64_t nextlen,
  const int64_t* nextcarry) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = offsets[i];
    int64_t stop = offsets[i + 1];
    int64_t count = stop - start;

    if (starts[parents[i]] == i) {
      for (int64_t k = 0;  k < maxcount;  k++) {
        nummissing[k] = 0;
      }
    }

    for (int64_t k = count;  k < maxcount;  k++) {
      nummissing[k]++;
    }

    for (int64_t j = 0;  j < count;  j++) {
      missing[start + j] = nummissing[j];
    }
  }

  for (int64_t j = 0;  j < nextlen;  j++) {
    nextshifts[j] = missing[nextcarry[j]];
  }
  return success();
}
