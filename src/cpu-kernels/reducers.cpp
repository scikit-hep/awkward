// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <vector>

#include "awkward/cpu-kernels/reducers.h"

template <typename T>
ERROR awkward_reduce_prod(T* toptr, const T* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (T)1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] *= fromptr[fromptroffset + i];
  }
  return success();
}
ERROR awkward_reduce_prod_bool_64(bool* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (bool)1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] *= (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_prod_int8_64(int8_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint8_64(uint8_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int16_64(int16_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint16_64(uint16_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}

ERROR awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toparents[i] = 0;
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_global_startstop_64(int64_t* globalstart, int64_t* globalstop, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  *globalstart = offsets[offsetsoffset + 0];
  *globalstop = offsets[offsetsoffset + length];
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

ERROR awkward_listoffsetarray_reduce_local_nextparents_64(int64_t* nextparents, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = offsets[offsetsoffset + i];  j < offsets[offsetsoffset + i + 1];  j++) {
      nextparents[j] = i;
    }
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_local_outoffsets_64(int64_t* outoffsets, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  outoffsets[outlength] = lenparents;
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    while (last < parents[parentsoffset + i]) {
      outoffsets[k] = i;
      k++;
      last++;
    }
  }
  return success();
}
