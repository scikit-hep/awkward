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

ERROR awkward_listoffsetarray_reduce_nextlen_64(int64_t* nextlen, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  *nextlen = offsets[offsetsoffset + i + 1] - offsets[offsetsoffset + i];
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_maxcount_offsetscopy_64(int64_t* maxcount, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  *maxcount = 0;
  offsetscopy[0] = offsets[offsetsoffset + 0];
  for (int64_t i = 0;  i < length;  i++) {
    int64_t count = offsets[offsetsoffset + i + 1] - offsets[offsetsoffset + i];
    if (*maxcount < count) {
      *maxcount = count;
    }
    offsetscopy[i + 1] = offsets[offsetsoffset + i + 1];
  }
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

ERROR awkward_listoffsetarray_reduce_nonlocal_findgaps_64(int64_t* gaps, const int64_t* parents, int64_t parentsoffset, int64_t lenparents) {
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[parentsoffset + i];
    if (last < parent) {
      gaps[k] = parent - last;
      k++;
      last = parent;
    }
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_outstartsstops_64(int64_t* outstarts, int64_t* outstops, const int64_t* distincts, int64_t lendistincts, const int64_t* gaps) {
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
  return success();
}
