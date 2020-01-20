// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/operations.h"

template <typename T, typename C>
ERROR awkward_listarray_count(T* tocount, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    tocount[i] = fromstops[stopsoffset + i] - fromstarts[startsoffset + i];
  }
  return success();
}
ERROR awkward_listarray32_count(int32_t* tocount, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_count<int32_t, int32_t>(tocount, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarrayU32_count(uint32_t* tocount, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_count<uint32_t, uint32_t>(tocount, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarray64_count(int64_t* tocount, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_count<int64_t, int64_t>(tocount, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarray32_count_64(int64_t* tocount, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_count<int64_t, int32_t>(tocount, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarrayU32_count_64(int64_t* tocount, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_count<int64_t, uint32_t>(tocount, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarray64_count_64(int64_t* tocount, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_count<int64_t, int64_t>(tocount, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}

ERROR awkward_regulararray_count(int64_t* tocount, int64_t size, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tocount[i] = size;
  }
  return success();
}

template <typename C>
ERROR awkward_indexedarray_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const C* fromindex, int64_t lenindex, int64_t indexoffset) {
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[indexoffset + i];
    if (j >= lencontent) {
      return failure("IndexedArray index out of range", i, j);
    }
    else if (j < 0) {
      tocount[i] = 0;
    }
    else {
      tocount[i] = contentcount[j];
    }
  }
  return success();
}
ERROR awkward_indexedarray32_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const int32_t* fromindex, int64_t lenindex, int64_t indexoffset) {
  return awkward_indexedarray_count<int32_t>(tocount, contentcount, lencontent, fromindex, lenindex, indexoffset);
}
ERROR awkward_indexedarrayU32_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const uint32_t* fromindex, int64_t lenindex, int64_t indexoffset) {
  return awkward_indexedarray_count<uint32_t>(tocount, contentcount, lencontent, fromindex, lenindex, indexoffset);
}
ERROR awkward_indexedarray64_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const int64_t* fromindex, int64_t lenindex, int64_t indexoffset) {
  return awkward_indexedarray_count<int64_t>(tocount, contentcount, lencontent, fromindex, lenindex, indexoffset);
}

template <typename C>
ERROR awkward_listarray_flatten_length(int64_t* tolen, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  *tolen = 0;
  for (int64_t i = 0; i < lenstarts; i++) {
    int64_t start = (C)fromstarts[startsoffset + i];
    int64_t stop = (C)fromstops[stopsoffset + i];
    if (start < 0  ||  stop < 0) {
      return failure("all start and stop values must be non-negative", kSliceNone, i);
    }
    int64_t length = stop - start;
    *tolen += length;
  }
  return success();
}
ERROR awkward_listarray32_flatten_length(int64_t* tolen, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten_length<int32_t>(tolen, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarrayU32_flatten_length(int64_t* tolen, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten_length<uint32_t>(tolen, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarray64_flatten_length(int64_t* tolen, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten_length<int64_t>(tolen, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}

template <typename C, typename T>
ERROR awkward_listarray_flatten(T* tocarry, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  int64_t at = 0;
  for (int64_t i = 0; i < lenstarts; i++) {
    int64_t start = (C)fromstarts[startsoffset + i];
    int64_t stop = (C)fromstops[stopsoffset + i];
    if (start < 0 || stop < 0) {
      return failure("all start and stop values must be non-negative", kSliceNone, i);
    }
    int64_t length = stop - start;
    if (length > 0) {
      for(int64_t l = 0; l < length; l++) {
        tocarry[at] = start + l;
        ++at;
      }
    }
  }
  return success();
}
ERROR awkward_listarray32_flatten_64(int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten<int32_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarrayU32_flatten_64(int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten<uint32_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarray64_flatten_64(int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten<int64_t, int64_t>(tocarry, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}

template <typename C, typename T>
ERROR awkward_indexedarray_flatten_nextcarry(T* tocarry, const C* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[indexoffset + i];
    if (j >= lencontent) {
      return failure("IndexedOptionArray index out of range", i, j);
    }
    else if (j >= 0) {
      tocarry[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_indexedarray32_flatten_nextcarry_64(int64_t* tocarry, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_flatten_nextcarry<int32_t, int64_t>(tocarry, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarrayU32_flatten_nextcarry_64(int64_t* tocarry, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_flatten_nextcarry<uint32_t, int64_t>(tocarry, fromindex, indexoffset, lenindex, lencontent);
}
ERROR awkward_indexedarray64_flatten_nextcarry_64(int64_t* tocarry, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  return awkward_indexedarray_flatten_nextcarry<int64_t, int64_t>(tocarry, fromindex, indexoffset, lenindex, lencontent);
}

template <typename C, typename M>
ERROR awkward_indexedarray_andmask(C* toindex, const M* mask, int64_t maskoffset, const C* fromindex, int64_t indexoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    M m = mask[maskoffset + i];
    if (m) {
      toindex[i] = -1;
    }
    else {
      toindex[i] = fromindex[indexoffset + i];
    }
  }
  return success();
}
ERROR awkward_indexedarray32_andmask_8(int32_t* toindex, const int8_t* mask, int64_t maskoffset, const int32_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_andmask<int32_t, int8_t>(toindex, mask, maskoffset, fromindex, indexoffset, length);
}
ERROR awkward_indexedarrayU32_andmask_8(uint32_t* toindex, const int8_t* mask, int64_t maskoffset, const uint32_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_andmask<uint32_t, int8_t>(toindex, mask, maskoffset, fromindex, indexoffset, length);
}
ERROR awkward_indexedarray64_andmask_8(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const int64_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_andmask<int64_t, int8_t>(toindex, mask, maskoffset, fromindex, indexoffset, length);
}
