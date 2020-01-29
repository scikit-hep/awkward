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
      return failure("index out of range", i, j);
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
ERROR awkward_listarray_flatten_scale(C* tostarts, C* tostops, const T* scale, const C* fromstarts, const C* fromstops,  int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  for (int64_t i = 0; i < lenstarts; i++) {
    int64_t start = (C)fromstarts[startsoffset + i];
    int64_t stop = (C)fromstops[stopsoffset + i];
    if (start < 0  ||  stop < 0) {
      return failure("all start and stop values must be non-negative", kSliceNone, i);
    }
    tostarts[i] = start * scale[i];
    tostops[i] = stop * scale[i];
  }
  return success();
}
ERROR awkward_listarray32_flatten_scale_64(int32_t* tostarts, int32_t* tostops, const int64_t* scale, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten_scale<int32_t, int64_t>(tostarts, tostops, scale, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarrayU32_flatten_scale_64(uint32_t* tostarts, uint32_t* tostops, const int64_t* scale, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten_scale<uint32_t, int64_t>(tostarts, tostops, scale, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_listarray64_flatten_scale_64(int64_t* tostarts, int64_t* tostops, const int64_t* scale, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_listarray_flatten_scale<int64_t, int64_t>(tostarts, tostops, scale, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}

template <typename C, typename T>
ERROR awkward_indexedarray_flatten_nextcarry(T* tocarry, const C* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[indexoffset + i];
    if (j >= lencontent) {
      return failure("index out of range", i, j);
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

template <typename T>
ERROR awkward_regulararray_compact_offsets(T* tooffsets, int64_t length, int64_t size) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[i + 1] = (i + 1)*size;
  }
  return success();
}
ERROR awkward_regulararray_compact_offsets64(int64_t* tooffsets, int64_t length, int64_t size) {
  return awkward_regulararray_compact_offsets<int64_t>(tooffsets, length, size);
}

template <typename C, typename T>
ERROR awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[startsoffset + i];
    C stop = fromstops[stopsoffset + i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone);
    }
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
  }
  return success();
}
ERROR awkward_listarray32_compact_offsets64(int64_t* tooffsets, const int32_t* fromstarts, const int32_t* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_compact_offsets<int32_t, int64_t>(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length);
}
ERROR awkward_listarrayU32_compact_offsets64(int64_t* tooffsets, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_compact_offsets<uint32_t, int64_t>(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length);
}
ERROR awkward_listarray64_compact_offsets64(int64_t* tooffsets, const int64_t* fromstarts, const int64_t* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_compact_offsets<int64_t, int64_t>(tooffsets, fromstarts, fromstops, startsoffset, stopsoffset, length);
}

template <typename C, typename T>
ERROR awkward_listoffsetarray_compact_offsets(T* tooffsets, const C* fromoffsets, int64_t offsetsoffset, int64_t length) {
  int64_t diff = (int64_t)fromoffsets[0];
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[i + 1] = fromoffsets[offsetsoffset + i + 1] - diff;
  }
  return success();
}
ERROR awkward_listoffsetarray32_compact_offsets64(int64_t* tooffsets, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t length) {
  return awkward_listoffsetarray_compact_offsets<int32_t, int64_t>(tooffsets, fromoffsets, offsetsoffset, length);
}
ERROR awkward_listoffsetarrayU32_compact_offsets64(int64_t* tooffsets, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t length) {
  return awkward_listoffsetarray_compact_offsets<uint32_t, int64_t>(tooffsets, fromoffsets, offsetsoffset, length);
}
ERROR awkward_listoffsetarray64_compact_offsets64(int64_t* tooffsets, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t length) {
  return awkward_listoffsetarray_compact_offsets<int64_t, int64_t>(tooffsets, fromoffsets, offsetsoffset, length);
}

template <typename C, typename T>
ERROR awkward_listarray_broadcast_tooffsets(T* tocarry, const T* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const C* fromstarts, int64_t startsoffset, const C* fromstops, int64_t stopsoffset, int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t start = (int64_t)fromstarts[startsoffset + i];
    int64_t stop = (int64_t)fromstops[stopsoffset + i];
    if (start != stop  &&  stop > lencontent) {
      return failure("stops[i] > len(content)", i, stop);
    }
    int64_t count = (int64_t)(fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone);
    }
    if (stop - start != count) {
      return failure("cannot broadcast nested list", i, kSliceNone);
    }
    for (int64_t j = start;  j < stop;  j++) {
      tocarry[k] = (T)j;
      k++;
    }
  }
  return success();
}
ERROR awkward_listarray32_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const int32_t* fromstarts, int64_t startsoffset, const int32_t* fromstops, int64_t stopsoffset, int64_t lencontent) {
  return awkward_listarray_broadcast_tooffsets<int32_t, int64_t>(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent);
}
ERROR awkward_listarrayU32_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const uint32_t* fromstarts, int64_t startsoffset, const uint32_t* fromstops, int64_t stopsoffset, int64_t lencontent) {
  return awkward_listarray_broadcast_tooffsets<uint32_t, int64_t>(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent);
}
ERROR awkward_listarray64_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const int64_t* fromstarts, int64_t startsoffset, const int64_t* fromstops, int64_t stopsoffset, int64_t lencontent) {
  return awkward_listarray_broadcast_tooffsets<int64_t, int64_t>(tocarry, fromoffsets, offsetsoffset, offsetslength, fromstarts, startsoffset, fromstops, stopsoffset, lencontent);
}

template <typename T>
ERROR awkward_regulararray_broadcast_tooffsets(const T* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, int64_t size) {
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)(fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone);
    }
    if (size != count) {
      return failure("cannot broadcast nested list", i, kSliceNone);
    }
  }
  return success();
}
ERROR awkward_regulararray_broadcast_tooffsets64(const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, int64_t size) {
  return awkward_regulararray_broadcast_tooffsets<int64_t>(fromoffsets, offsetsoffset, offsetslength, size);
}

template <typename T>
ERROR awkward_regulararray_broadcast_tooffsets_size1(T* tocarry, const T* fromoffsets, int64_t offsetsoffset, int64_t offsetslength) {
  int64_t k = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)(fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone);
    }
    for (int64_t j = 0;  j < count;  j++) {
      tocarry[k] = (T)i;
      k++;
    }
  }
  return success();
}
ERROR awkward_regulararray_broadcast_tooffsets64_size1(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_regulararray_broadcast_tooffsets_size1<int64_t>(tocarry, fromoffsets, offsetsoffset, offsetslength);
}

template <typename C>
ERROR awkward_listoffsetarray_toRegularArray(int64_t* size, const C* fromoffsets, int64_t offsetsoffset, int64_t offsetslength) {
  *size = -1;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)(fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i]);
    if (count < 0) {
      return failure("offsets must be monotonically increasing", i, kSliceNone);
    }
    if (*size == -1) {
      *size = count;
    }
    else if (*size != count) {
      return failure("cannot convert to RegularArray because subarray lengths are not regular", i, kSliceNone);
    }
  }
  if (*size == -1) {
    *size = 0;
  }
  return success();
}
ERROR awkward_listoffsetarray32_toRegularArray(int64_t* size, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_listoffsetarray_toRegularArray<int32_t>(size, fromoffsets, offsetsoffset, offsetslength);
}
ERROR awkward_listoffsetarrayU32_toRegularArray(int64_t* size, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_listoffsetarray_toRegularArray<uint32_t>(size, fromoffsets, offsetsoffset, offsetslength);
}
ERROR awkward_listoffsetarray64_toRegularArray(int64_t* size, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_listoffsetarray_toRegularArray<int64_t>(size, fromoffsets, offsetsoffset, offsetslength);
}
