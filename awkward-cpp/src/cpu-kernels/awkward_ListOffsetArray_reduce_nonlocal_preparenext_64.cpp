// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_preparenext_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t nextlen,
  int64_t* maxnextparents,
  int64_t* distincts,
  int64_t distinctslen,
  int64_t* offsetscopy,
  const int64_t* offsets,
  int64_t length,
  const int64_t* parents,
  int64_t maxcount) {
  *maxnextparents = 0;
  for (int64_t i = 0;  i < distinctslen;  i++) {
    distincts[i] = -1;
  }

  int64_t k = 0;
  while (k < nextlen) {
    int64_t j = 0;
    for (int64_t i = 0;  i < length;  i++) {
      if (offsetscopy[i] < offsets[i + 1]) {
        int64_t diff = offsetscopy[i] - offsets[i];
        int64_t parent = parents[i];

        nextcarry[k] = offsetscopy[i];
        nextparents[k] = parent*maxcount + diff;

        if (*maxnextparents < nextparents[k]) {
          *maxnextparents = nextparents[k];
        }

        if (distincts[nextparents[k]] == -1) {
          distincts[nextparents[k]] = j;
          j++;
        }

        k++;
        offsetscopy[i]++;
      }
    }
  }
  return success();
}
