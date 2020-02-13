// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <vector>

#include "awkward/cpu-kernels/reducers.h"

ERROR awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toparents[i] = 0;
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_maxcount_offsetscopy_64(int64_t* maxcount, int64_t* offsetscopy, int64_t* nextlen, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  *maxcount = 0;
  offsetscopy[0] = offsets[offsetsoffset + 0];
  for (int64_t i = 0;  i < length;  i++) {
    int64_t count = offsets[offsetsoffset + i + 1] - offsets[offsetsoffset + i];
    if (*maxcount < count) {
      *maxcount = count;
    }
    offsetscopy[i + 1] = offsets[offsetsoffset + i + 1];
  }
  *nextlen = offsetscopy[length] - offsetscopy[0];
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_preparenext_64(int64_t* nextcarry, int64_t* nextparents, int64_t nextlen, int64_t* maxnextparents, int64_t* distincts, int64_t distinctslen, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length, const int64_t* parents, int64_t parentsoffset, int64_t maxcount) {
  *maxnextparents = 0;
  for (int64_t i = 0;  i < distinctslen;  i++) {
    distincts[i] = -1;
  }

  int64_t k = 0;
  while (k < nextlen) {
    int64_t j = 0;
    for (int64_t i = 0;  i < length;  i++) {
      if (offsetscopy[i] < offsets[offsetsoffset + i + 1]) {
        int64_t count = offsets[offsetsoffset + i + 1] - offsets[offsetsoffset + i];
        int64_t diff = offsetscopy[i] - offsets[offsetsoffset + i];

        nextcarry[k] = offsetscopy[i];
        nextparents[k] = parents[parentsoffset + i]*maxcount + diff;
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
