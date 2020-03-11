// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/operations.h"

template <typename T, typename C>
ERROR awkward_listarray_num(T* tonum, const C* fromstarts, int64_t startsoffset, const C* fromstops, int64_t stopsoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[startsoffset + i];
    C stop = fromstops[stopsoffset + i];
    tonum[i] = (T)(stop - start);
  }
  return success();
}
ERROR awkward_listarray32_num_64(int64_t* tonum, const int32_t* fromstarts, int64_t startsoffset, const int32_t* fromstops, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_num<int64_t, int32_t>(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length);
}
ERROR awkward_listarrayU32_num_64(int64_t* tonum, const uint32_t* fromstarts, int64_t startsoffset, const uint32_t* fromstops, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_num<int64_t, uint32_t>(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length);
}
ERROR awkward_listarray64_num_64(int64_t* tonum, const int64_t* fromstarts, int64_t startsoffset, const int64_t* fromstops, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_num<int64_t, int64_t>(tonum, fromstarts, startsoffset, fromstops, stopsoffset, length);
}

template <typename T>
ERROR awkward_regulararray_num(T* tonum, int64_t size, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tonum[i] = size;
  }
  return success();
}
ERROR awkward_regulararray_num_64(int64_t* tonum, int64_t size, int64_t length) {
  return awkward_regulararray_num<int64_t>(tonum, size, length);
}

template <typename T, typename C>
ERROR awkward_listoffsetarray_flatten_offsets(T* tooffsets, const C* outeroffsets, int64_t outeroffsetsoffset, int64_t outeroffsetslen, const T* inneroffsets, int64_t inneroffsetsoffset, int64_t inneroffsetslen) {
  for (int64_t i = 0;  i < outeroffsetslen;  i++) {
    tooffsets[i] = inneroffsets[inneroffsetsoffset + outeroffsets[outeroffsetsoffset + i]];
  }
  return success();
}
ERROR awkward_listoffsetarray32_flatten_offsets_64(int64_t* tooffsets, const int32_t* outeroffsets, int64_t outeroffsetsoffset, int64_t outeroffsetslen, const int64_t* inneroffsets, int64_t inneroffsetsoffset, int64_t inneroffsetslen) {
  return awkward_listoffsetarray_flatten_offsets<int64_t, int32_t>(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen);
}
ERROR awkward_listoffsetarrayU32_flatten_offsets_64(int64_t* tooffsets, const uint32_t* outeroffsets, int64_t outeroffsetsoffset, int64_t outeroffsetslen, const int64_t* inneroffsets, int64_t inneroffsetsoffset, int64_t inneroffsetslen) {
  return awkward_listoffsetarray_flatten_offsets<int64_t, uint32_t>(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen);
}
ERROR awkward_listoffsetarray64_flatten_offsets_64(int64_t* tooffsets, const int64_t* outeroffsets, int64_t outeroffsetsoffset, int64_t outeroffsetslen, const int64_t* inneroffsets, int64_t inneroffsetsoffset, int64_t inneroffsetslen) {
  return awkward_listoffsetarray_flatten_offsets<int64_t, int64_t>(tooffsets, outeroffsets, outeroffsetsoffset, outeroffsetslen, inneroffsets, inneroffsetsoffset, inneroffsetslen);
}

template <typename T, typename C>
ERROR awkward_indexedarray_flatten_none2empty(T* outoffsets, const C* outindex, int64_t outindexoffset, int64_t outindexlength, const T* offsets, int64_t offsetsoffset, int64_t offsetslength) {
  outoffsets[0] = offsets[offsetsoffset + 0];
  int64_t k = 1;
  for (int64_t i = 0;  i < outindexlength;  i++) {
    C idx = outindex[outindexoffset + i];
    if (idx < 0) {
      outoffsets[k] = outoffsets[k - 1];
      k++;
    }
    else if (offsetsoffset + idx + 1 >= offsetslength) {
      return failure("flattening offset out of range", i, kSliceNone);
    }
    else {
      T count = offsets[offsetsoffset + idx + 1] - offsets[offsetsoffset + idx];
      outoffsets[k] = outoffsets[k - 1] + count;
      k++;
    }
  }
  return success();
}
ERROR awkward_indexedarray32_flatten_none2empty_64(int64_t* outoffsets, const int32_t* outindex, int64_t outindexoffset, int64_t outindexlength, const int64_t* offsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_indexedarray_flatten_none2empty<int64_t, int32_t>(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength);
}
ERROR awkward_indexedarrayU32_flatten_none2empty_64(int64_t* outoffsets, const uint32_t* outindex, int64_t outindexoffset, int64_t outindexlength, const int64_t* offsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_indexedarray_flatten_none2empty<int64_t, uint32_t>(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength);
}
ERROR awkward_indexedarray64_flatten_none2empty_64(int64_t* outoffsets, const int64_t* outindex, int64_t outindexoffset, int64_t outindexlength, const int64_t* offsets, int64_t offsetsoffset, int64_t offsetslength) {
  return awkward_indexedarray_flatten_none2empty<int64_t, int64_t>(outoffsets, outindex, outindexoffset, outindexlength, offsets, offsetsoffset, offsetslength);
}

template <typename FROMTAGS, typename FROMINDEX, typename T>
ERROR awkward_unionarray_flatten_length(int64_t* total_length, const FROMTAGS* fromtags, int64_t fromtagsoffset, const FROMINDEX* fromindex, int64_t fromindexoffset, int64_t length, T** offsetsraws, int64_t* offsetsoffsets) {
  *total_length = 0;
  for (int64_t i = 0;  i < length;  i++) {
    FROMTAGS tag = fromtags[fromtagsoffset + i];
    FROMINDEX idx = fromindex[fromindexoffset + i];
    T start = offsetsraws[tag][offsetsoffsets[tag] + idx];
    T stop = offsetsraws[tag][offsetsoffsets[tag] + idx + 1];
    *total_length = *total_length + (stop - start);
  }
  return success();
}
ERROR awkward_unionarray32_flatten_length_64(int64_t* total_length, const int8_t* fromtags, int64_t fromtagsoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t** offsetsraws, int64_t* offsetsoffsets) {
  return awkward_unionarray_flatten_length<int8_t, int32_t, int64_t>(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets);
}
ERROR awkward_unionarrayU32_flatten_length_64(int64_t* total_length, const int8_t* fromtags, int64_t fromtagsoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t** offsetsraws, int64_t* offsetsoffsets) {
  return awkward_unionarray_flatten_length<int8_t, uint32_t, int64_t>(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets);
}
ERROR awkward_unionarray64_flatten_length_64(int64_t* total_length, const int8_t* fromtags, int64_t fromtagsoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t** offsetsraws, int64_t* offsetsoffsets) {
  return awkward_unionarray_flatten_length<int8_t, int64_t, int64_t>(total_length, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets);
}

template <typename FROMTAGS, typename FROMINDEX, typename TOTAGS, typename TOINDEX, typename T>
ERROR awkward_unionarray_flatten_combine(TOTAGS* totags, TOINDEX* toindex, T* tooffsets, const FROMTAGS* fromtags, int64_t fromtagsoffset, const FROMINDEX* fromindex, int64_t fromindexoffset, int64_t length, T** offsetsraws, int64_t* offsetsoffsets) {
  tooffsets[0] = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    FROMTAGS tag = fromtags[fromtagsoffset + i];
    FROMINDEX idx = fromindex[fromindexoffset + i];
    T start = offsetsraws[tag][offsetsoffsets[tag] + idx];
    T stop = offsetsraws[tag][offsetsoffsets[tag] + idx + 1];
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
    for (int64_t j = start;  j < stop;  j++) {
      totags[k] = tag;
      toindex[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_unionarray32_flatten_combine_64(int8_t* totags, int64_t* toindex, int64_t* tooffsets, const int8_t* fromtags, int64_t fromtagsoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t** offsetsraws, int64_t* offsetsoffsets) {
  return awkward_unionarray_flatten_combine<int8_t, int32_t, int8_t, int64_t, int64_t>(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets);
}
ERROR awkward_unionarrayU32_flatten_combine_64(int8_t* totags, int64_t* toindex, int64_t* tooffsets, const int8_t* fromtags, int64_t fromtagsoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t** offsetsraws, int64_t* offsetsoffsets) {
  return awkward_unionarray_flatten_combine<int8_t, uint32_t, int8_t, int64_t, int64_t>(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets);
}
ERROR awkward_unionarray64_flatten_combine_64(int8_t* totags, int64_t* toindex, int64_t* tooffsets, const int8_t* fromtags, int64_t fromtagsoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t** offsetsraws, int64_t* offsetsoffsets) {
  return awkward_unionarray_flatten_combine<int8_t, int64_t, int8_t, int64_t, int64_t>(totags, toindex, tooffsets, fromtags, fromtagsoffset, fromindex, fromindexoffset, length, offsetsraws, offsetsoffsets);
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

template <typename C, typename M, typename TO>
ERROR awkward_indexedarray_overlay_mask(TO* toindex, const M* mask, int64_t maskoffset, const C* fromindex, int64_t indexoffset, int64_t length) {
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
ERROR awkward_indexedarray32_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const int32_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_overlay_mask<int32_t, int8_t, int64_t>(toindex, mask, maskoffset, fromindex, indexoffset, length);
}
ERROR awkward_indexedarrayU32_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const uint32_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_overlay_mask<uint32_t, int8_t, int64_t>(toindex, mask, maskoffset, fromindex, indexoffset, length);
}
ERROR awkward_indexedarray64_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const int64_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_overlay_mask<int64_t, int8_t, int64_t>(toindex, mask, maskoffset, fromindex, indexoffset, length);
}

template <typename C, typename M>
ERROR awkward_indexedarray_mask(M* tomask, const C* fromindex, int64_t indexoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = (fromindex[indexoffset + i] < 0);
  }
  return success();
}
ERROR awkward_indexedarray32_mask8(int8_t* tomask, const int32_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_mask<int32_t, int8_t>(tomask, fromindex, indexoffset, length);
}
ERROR awkward_indexedarrayU32_mask8(int8_t* tomask, const uint32_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_mask<uint32_t, int8_t>(tomask, fromindex, indexoffset, length);
}
ERROR awkward_indexedarray64_mask8(int8_t* tomask, const int64_t* fromindex, int64_t indexoffset, int64_t length) {
  return awkward_indexedarray_mask<int64_t, int8_t>(tomask, fromindex, indexoffset, length);
}

template <typename M>
ERROR awkward_zero_mask(M* tomask, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = 0;
  }
  return success();
}
ERROR awkward_zero_mask8(int8_t* tomask, int64_t length) {
  return awkward_zero_mask<int8_t>(tomask, length);
}

template <typename OUT, typename IN, typename TO>
ERROR awkward_indexedarray_simplify(TO* toindex, const OUT* outerindex, int64_t outeroffset, int64_t outerlength, const IN* innerindex, int64_t inneroffset, int64_t innerlength) {
  for (int64_t i = 0;  i < outerlength;  i++) {
    OUT j = outerindex[outeroffset + i];
    if (j < 0) {
      toindex[i] = -1;
    }
    else if (j >= innerlength) {
      return failure("index out of range", i, j);
    }
    else {
      toindex[i] = innerindex[inneroffset + j];
    }
  }
  return success();
}
ERROR awkward_indexedarray32_simplify32_to64(int64_t* toindex, const int32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<int32_t, int32_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarray32_simplifyU32_to64(int64_t* toindex, const int32_t* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<int32_t, uint32_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarray32_simplify64_to64(int64_t* toindex, const int32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<int32_t, int64_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarrayU32_simplify32_to64(int64_t* toindex, const uint32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<uint32_t, int32_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarrayU32_simplifyU32_to64(int64_t* toindex, const uint32_t* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<uint32_t, uint32_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarrayU32_simplify64_to64(int64_t* toindex, const uint32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<uint32_t, int64_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarray64_simplify32_to64(int64_t* toindex, const int64_t* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<int64_t, int32_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarray64_simplifyU32_to64(int64_t* toindex, const int64_t* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<int64_t, uint32_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
}
ERROR awkward_indexedarray64_simplify64_to64(int64_t* toindex, const int64_t* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength) {
  return awkward_indexedarray_simplify<int64_t, int64_t, int64_t>(toindex, outerindex, outeroffset, outerlength, innerindex, inneroffset, innerlength);
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
  int64_t diff = (int64_t)fromoffsets[offsetsoffset + 0];
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

template <typename FROM, typename TO>
ERROR awkward_numpyarray_fill(TO* toptr, int64_t tooffset, const FROM* fromptr, int64_t fromoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[tooffset + i] = (TO)fromptr[fromoffset + i];
  }
  return success();
}
template <typename TO>
ERROR awkward_numpyarray_fill_frombool(TO* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[tooffset + i] = (TO)(fromptr[fromoffset + i] != 0);
  }
  return success();
}
ERROR awkward_numpyarray_fill_todouble_fromdouble(double* toptr, int64_t tooffset, const double* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<double, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_fromfloat(double* toptr, int64_t tooffset, const float* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<float, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_from64(double* toptr, int64_t tooffset, const int64_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int64_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_fromU64(double* toptr, int64_t tooffset, const uint64_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint64_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_from32(double* toptr, int64_t tooffset, const int32_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int32_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_fromU32(double* toptr, int64_t tooffset, const uint32_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint32_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_from16(double* toptr, int64_t tooffset, const int16_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int16_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_fromU16(double* toptr, int64_t tooffset, const uint16_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint16_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_from8(double* toptr, int64_t tooffset, const int8_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int8_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_fromU8(double* toptr, int64_t tooffset, const uint8_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint8_t, double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_todouble_frombool(double* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill_frombool<double>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_toU64_fromU64(uint64_t* toptr, int64_t tooffset, const uint64_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint64_t, uint64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_from64(int64_t* toptr, int64_t tooffset, const int64_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int64_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_fromU64(int64_t* toptr, int64_t tooffset, const uint64_t* fromptr, int64_t fromoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    if (fromptr[fromoffset + i] > kMaxInt64) {
      return failure("uint64 value too large for int64 output", i, kSliceNone);
    }
    toptr[tooffset + i] = fromptr[fromoffset + i];
  }
  return success();
}
ERROR awkward_numpyarray_fill_to64_from32(int64_t* toptr, int64_t tooffset, const int32_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int32_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_fromU32(int64_t* toptr, int64_t tooffset, const uint32_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint32_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_from16(int64_t* toptr, int64_t tooffset, const int16_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int16_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_fromU16(int64_t* toptr, int64_t tooffset, const uint16_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint16_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_from8(int64_t* toptr, int64_t tooffset, const int8_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<int8_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_fromU8(int64_t* toptr, int64_t tooffset, const uint8_t* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill<uint8_t, int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_to64_frombool(int64_t* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill_frombool<int64_t>(toptr, tooffset, fromptr, fromoffset, length);
}
ERROR awkward_numpyarray_fill_tobool_frombool(bool* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length) {
  return awkward_numpyarray_fill_frombool<bool>(toptr, tooffset, fromptr, fromoffset, length);
}

template <typename FROM, typename TO>
ERROR awkward_listarray_fill(TO* tostarts, int64_t tostartsoffset, TO* tostops, int64_t tostopsoffset, const FROM* fromstarts, int64_t fromstartsoffset, const FROM* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    tostarts[tostartsoffset + i] = (TO)(fromstarts[fromstartsoffset + i] + base);
    tostops[tostopsoffset + i] = (TO)(fromstops[fromstopsoffset + i] + base);
  }
  return success();
}
ERROR awkward_listarray_fill_to64_from32(int64_t* tostarts, int64_t tostartsoffset, int64_t* tostops, int64_t tostopsoffset, const int32_t* fromstarts, int64_t fromstartsoffset, const int32_t* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base) {
  return awkward_listarray_fill<int32_t, int64_t>(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base);
}
ERROR awkward_listarray_fill_to64_fromU32(int64_t* tostarts, int64_t tostartsoffset, int64_t* tostops, int64_t tostopsoffset, const uint32_t* fromstarts, int64_t fromstartsoffset, const uint32_t* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base) {
  return awkward_listarray_fill<uint32_t, int64_t>(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base);
}
ERROR awkward_listarray_fill_to64_from64(int64_t* tostarts, int64_t tostartsoffset, int64_t* tostops, int64_t tostopsoffset, const int64_t* fromstarts, int64_t fromstartsoffset, const int64_t* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base) {
  return awkward_listarray_fill<int64_t, int64_t>(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstartsoffset, fromstops, fromstopsoffset, length, base);
}

template <typename FROM, typename TO>
ERROR awkward_indexedarray_fill(TO* toindex, int64_t toindexoffset, const FROM* fromindex, int64_t fromindexoffset, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    FROM from = fromindex[fromindexoffset + i];
    toindex[toindexoffset + i] = from < 0 ? -1 : (TO)(from + base);
  }
  return success();
}
ERROR awkward_indexedarray_fill_to64_from32(int64_t* toindex, int64_t toindexoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t base) {
  return awkward_indexedarray_fill<int32_t, int64_t>(toindex, toindexoffset, fromindex, fromindexoffset, length, base);
}
ERROR awkward_indexedarray_fill_to64_fromU32(int64_t* toindex, int64_t toindexoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t base) {
  return awkward_indexedarray_fill<uint32_t, int64_t>(toindex, toindexoffset, fromindex, fromindexoffset, length, base);
}
ERROR awkward_indexedarray_fill_to64_from64(int64_t* toindex, int64_t toindexoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t base) {
  return awkward_indexedarray_fill<int64_t, int64_t>(toindex, toindexoffset, fromindex, fromindexoffset, length, base);
}

template <typename TO>
ERROR awkward_indexedarray_fill_count(TO* toindex, int64_t toindexoffset, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = i + base;
  }
  return success();
}
ERROR awkward_indexedarray_fill_to64_count(int64_t* toindex, int64_t toindexoffset, int64_t length, int64_t base) {
  return awkward_indexedarray_fill_count(toindex, toindexoffset, length, base);
}

template <typename FROM, typename TO>
ERROR awkward_unionarray_filltags(TO* totags, int64_t totagsoffset, const FROM* fromtags, int64_t fromtagsoffset, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    totags[totagsoffset + i] = (TO)(fromtags[fromtagsoffset + i] + base);
  }
  return success();
}
ERROR awkward_unionarray_filltags_to8_from8(int8_t* totags, int64_t totagsoffset, const int8_t* fromtags, int64_t fromtagsoffset, int64_t length, int64_t base) {
  return awkward_unionarray_filltags<int8_t, int8_t>(totags, totagsoffset, fromtags, fromtagsoffset, length, base);
}

template <typename FROM, typename TO>
ERROR awkward_unionarray_fillindex(TO* toindex, int64_t toindexoffset, const FROM* fromindex, int64_t fromindexoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = (TO)fromindex[fromindexoffset + i];
  }
  return success();
}
ERROR awkward_unionarray_fillindex_to64_from32(int64_t* toindex, int64_t toindexoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t length) {
  return awkward_unionarray_fillindex<int32_t, int64_t>(toindex, toindexoffset, fromindex, fromindexoffset, length);
}
ERROR awkward_unionarray_fillindex_to64_fromU32(int64_t* toindex, int64_t toindexoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t length) {
  return awkward_unionarray_fillindex<uint32_t, int64_t>(toindex, toindexoffset, fromindex, fromindexoffset, length);
}
ERROR awkward_unionarray_fillindex_to64_from64(int64_t* toindex, int64_t toindexoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t length) {
  return awkward_unionarray_fillindex<int64_t, int64_t>(toindex, toindexoffset, fromindex, fromindexoffset, length);
}

template <typename TO>
ERROR awkward_unionarray_filltags_const(TO* totags, int64_t totagsoffset, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    totags[totagsoffset + i] = (TO)base;
  }
  return success();
}
ERROR awkward_unionarray_filltags_to8_const(int8_t* totags, int64_t totagsoffset, int64_t length, int64_t base) {
  return awkward_unionarray_filltags_const<int8_t>(totags, totagsoffset, length, base);
}

template <typename TO>
ERROR awkward_unionarray_fillindex_count(TO* toindex, int64_t toindexoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = (TO)i;
  }
  return success();
}
ERROR awkward_unionarray_fillindex_to64_count(int64_t* toindex, int64_t toindexoffset, int64_t length) {
  return awkward_unionarray_fillindex_count<int64_t>(toindex, toindexoffset, length);
}

template <typename OUTERTAGS, typename OUTERINDEX, typename INNERTAGS, typename INNERINDEX, typename TOTAGS, typename TOINDEX>
ERROR awkward_unionarray_simplify(TOTAGS* totags, TOINDEX* toindex, const OUTERTAGS* outertags, int64_t outertagsoffset, const OUTERINDEX* outerindex, int64_t outerindexoffset, const INNERTAGS* innertags, int64_t innertagsoffset, const INNERINDEX* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    if (outertags[outertagsoffset + i] == outerwhich) {
      OUTERINDEX j = outerindex[outerindexoffset + i];
      if (innertags[innertagsoffset + j] == innerwhich) {
        totags[i] = (TOTAGS)towhich;
        toindex[i] = (TOINDEX)(innerindex[innerindexoffset + j] + base);
      }
    }
  }
  return success();
}
ERROR awkward_unionarray8_32_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, int32_t, int8_t, int32_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_32_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, int32_t, int8_t, uint32_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_32_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, int32_t, int8_t, int64_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_U32_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const uint32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, uint32_t, int8_t, int32_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_U32_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const uint32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, uint32_t, int8_t, uint32_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_U32_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const uint32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, uint32_t, int8_t, int64_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_64_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int64_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, int64_t, int8_t, int32_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_64_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int64_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, int64_t, int8_t, uint32_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}
ERROR awkward_unionarray8_64_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int64_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify<int8_t, int64_t, int8_t, int64_t, int8_t, int64_t>(totags, toindex, outertags, outertagsoffset, outerindex, outerindexoffset, innertags, innertagsoffset, innerindex, innerindexoffset, towhich, innerwhich, outerwhich, length, base);
}

template <typename FROMTAGS, typename FROMINDEX, typename TOTAGS, typename TOINDEX>
ERROR awkward_unionarray_simplify_one(TOTAGS* totags, TOINDEX* toindex, const FROMTAGS* fromtags, int64_t fromtagsoffset, const FROMINDEX* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    if (fromtags[fromtagsoffset + i] == fromwhich) {
      totags[i] = (TOTAGS)towhich;
      toindex[i] = (TOINDEX)(fromindex[fromindexoffset + i] + base);
    }
  }
  return success();
}
ERROR awkward_unionarray8_32_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const int8_t* fromtags, int64_t fromtagsoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify_one<int8_t, int32_t, int8_t, int64_t>(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base);
}
ERROR awkward_unionarray8_U32_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const int8_t* fromtags, int64_t fromtagsoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify_one<int8_t, uint32_t, int8_t, int64_t>(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base);
}
ERROR awkward_unionarray8_64_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const int8_t* fromtags, int64_t fromtagsoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base) {
  return awkward_unionarray_simplify_one<int8_t, int64_t, int8_t, int64_t>(totags, toindex, fromtags, fromtagsoffset, fromindex, fromindexoffset, towhich, fromwhich, length, base);
}

template <typename C>
ERROR awkward_listarray_validity(const C* starts, int64_t startsoffset, const C* stops, int64_t stopsoffset, int64_t length, int64_t lencontent) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = starts[startsoffset + i];
    C stop = stops[stopsoffset + i];
    if (start != stop) {
      if (start > stop) {
        return failure("start[i] > stop[i]", i, kSliceNone);
      }
      if (start < 0) {
        return failure("start[i] < 0", i, kSliceNone);
      }
      if (stop > lencontent) {
        return failure("stop[i] > len(content)", i, kSliceNone);
      }
    }
  }
  return success();
}

ERROR awkward_listarray32_validity(const int32_t* starts, int64_t startsoffset, const int32_t* stops, int64_t stopsoffset, int64_t length, int64_t lencontent) {
  return awkward_listarray_validity<int32_t>(starts, startsoffset, stops, stopsoffset, length, lencontent);
}
ERROR awkward_listarrayU32_validity(const uint32_t* starts, int64_t startsoffset, const uint32_t* stops, int64_t stopsoffset, int64_t length, int64_t lencontent) {
  return awkward_listarray_validity<uint32_t>(starts, startsoffset, stops, stopsoffset, length, lencontent);
}
ERROR awkward_listarray64_validity(const int64_t* starts, int64_t startsoffset, const int64_t* stops, int64_t stopsoffset, int64_t length, int64_t lencontent) {
  return awkward_listarray_validity<int64_t>(starts, startsoffset, stops, stopsoffset, length, lencontent);
}

template <typename C, bool ISOPTION>
ERROR awkward_indexedarray_validity(const C* index, int64_t indexoffset, int64_t length, int64_t lencontent) {
  for (int64_t i = 0;  i < length;  i++) {
    C idx = index[indexoffset + i];
    if (!ISOPTION) {
      if (idx < 0) {
        return failure("index[i] < 0", i, kSliceNone);
      }
    }
    if (idx >= lencontent) {
      return failure("index[i] >= len(content)", i, kSliceNone);
    }
  }
  return success();
}
ERROR awkward_indexedarray32_validity(const int32_t* index, int64_t indexoffset, int64_t length, int64_t lencontent, bool isoption) {
  if (isoption) {
    return awkward_indexedarray_validity<int32_t, true>(index, indexoffset, length, lencontent);
  }
  else {
    return awkward_indexedarray_validity<int32_t, false>(index, indexoffset, length, lencontent);
  }
}
ERROR awkward_indexedarrayU32_validity(const uint32_t* index, int64_t indexoffset, int64_t length, int64_t lencontent, bool isoption) {
  if (isoption) {
    return awkward_indexedarray_validity<uint32_t, true>(index, indexoffset, length, lencontent);
  }
  else {
    return awkward_indexedarray_validity<uint32_t, false>(index, indexoffset, length, lencontent);
  }
}
ERROR awkward_indexedarray64_validity(const int64_t* index, int64_t indexoffset, int64_t length, int64_t lencontent, bool isoption) {
  if (isoption) {
    return awkward_indexedarray_validity<int64_t, true>(index, indexoffset, length, lencontent);
  }
  else {
    return awkward_indexedarray_validity<int64_t, false>(index, indexoffset, length, lencontent);
  }
}

template <typename T, typename I>
ERROR awkward_unionarray_validity(const T* tags, int64_t tagsoffset, const I* index, int64_t indexoffset, int64_t length, int64_t numcontents, const int64_t* lencontents) {
  for (int64_t i = 0;  i < length;  i++) {
    T tag = tags[tagsoffset + i];
    I idx = index[indexoffset + i];
    if (tag < 0) {
      return failure("tags[i] < 0", i, kSliceNone);
    }
    if (idx < 0) {
      return failure("index[i] < 0", i, kSliceNone);
    }
    if (tag >= numcontents) {
      return failure("tags[i] >= len(contents)", i, kSliceNone);
    }
    int64_t lencontent = lencontents[tag];
    if (idx >= lencontent) {
      return failure("index[i] >= len(content[tags[i]])", i, kSliceNone);
    }
  }
  return success();
}
ERROR awkward_unionarray8_32_validity(const int8_t* tags, int64_t tagsoffset, const int32_t* index, int64_t indexoffset, int64_t length, int64_t numcontents, const int64_t* lencontents) {
  return awkward_unionarray_validity<int8_t, int32_t>(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents);
}
ERROR awkward_unionarray8_U32_validity(const int8_t* tags, int64_t tagsoffset, const uint32_t* index, int64_t indexoffset, int64_t length, int64_t numcontents, const int64_t* lencontents) {
  return awkward_unionarray_validity<int8_t, uint32_t>(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents);
}
ERROR awkward_unionarray8_64_validity(const int8_t* tags, int64_t tagsoffset, const int64_t* index, int64_t indexoffset, int64_t length, int64_t numcontents, const int64_t* lencontents) {
  return awkward_unionarray_validity<int8_t, int64_t>(tags, tagsoffset, index, indexoffset, length, numcontents, lencontents);
}

template <typename T>
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1(T* toindex, const int8_t* frommask, int64_t length) {
  int64_t count = 0;
  for (int64_t i = 0; i < length; i++) {
    if(frommask[i]) {
      toindex[i] = -1;
    }
    else {
      toindex[i] = count++;
    }
  }
  return success();
}
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(int64_t* toindex, const int8_t* frommask, int64_t length) {
  return awkward_IndexedOptionArray_rpad_and_clip_mask_axis1<int64_t>(toindex, frommask, length);
}

template <typename T>
ERROR awkward_index_rpad_and_clip_axis0(T* toindex, int64_t target, int64_t length) {
  int64_t shorter = (target < length ? target : length);
  for (int64_t i = 0; i < shorter; i++) {
    toindex[i] = i;
  }
  for (int64_t i = shorter; i < target; i++) {
    toindex[i] = -1;
  }
  return success();
}
ERROR awkward_index_rpad_and_clip_axis0_64(int64_t* toindex, int64_t target, int64_t length) {
  return awkward_index_rpad_and_clip_axis0<int64_t>(toindex, target, length);
}

template <typename T>
ERROR awkward_index_rpad_and_clip_axis1(T* tostarts, T* tostops, int64_t target, int64_t length) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    offset = offset + target;
    tostops[i] = offset;
   }
  return success();
}
ERROR awkward_index_rpad_and_clip_axis1_64(int64_t* tostarts, int64_t* tostops, int64_t target, int64_t length) {
  return awkward_index_rpad_and_clip_axis1<int64_t>(tostarts, tostops, target, length);
}

template <typename T>
ERROR awkward_RegularArray_rpad_and_clip_axis1(T* toindex, int64_t target, int64_t size, int64_t length) {
  int64_t shorter = (target < size ? target : size);
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < shorter;  j++) {
      toindex[i*target + j] = i*size + j;
    }
    for (int64_t j = shorter;  j < target;  j++) {
      toindex[i*target + j] = -1;
    }
  }
  return success();
}
ERROR awkward_RegularArray_rpad_and_clip_axis1_64(int64_t* toindex, int64_t target, int64_t size, int64_t length) {
  return awkward_RegularArray_rpad_and_clip_axis1<int64_t>(toindex, target, size, length);
}

template <typename C>
ERROR awkward_ListArray_min_range(int64_t* tomin, const C* fromstarts, const C* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  int64_t shorter = fromstops[stopsoffset + 0] - fromstarts[startsoffset + 0];
  for (int64_t i = 1;  i < lenstarts;  i++) {
    int64_t range = fromstops[startsoffset + i] - fromstarts[stopsoffset + i];
    shorter = (shorter < range) ? shorter : range;
  }
  *tomin = shorter;
  return success();
}
ERROR awkward_ListArray32_min_range(int64_t* tomin, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_min_range<int32_t>(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_ListArrayU32_min_range(int64_t* tomin, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_min_range<uint32_t>(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_ListArray64_min_range(int64_t* tomin, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_min_range<int64_t>(tomin, fromstarts, fromstops, lenstarts, startsoffset, stopsoffset);
}

template <typename C>
ERROR awkward_ListArray_rpad_and_clip_length_axis1(int64_t* tolength, const C* fromstarts, const C* fromstops, int64_t target, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  int64_t length = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t range = fromstops[startsoffset + i] - fromstarts[stopsoffset + i];
    length += (target > range) ? target : range;
  }
  *tolength = length;
  return success();
}
ERROR awkward_ListArray32_rpad_and_clip_length_axis1(int64_t* tomin, const int32_t* fromstarts, const int32_t* fromstops, int64_t target, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_rpad_and_clip_length_axis1<int32_t>(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_ListArrayU32_rpad_and_clip_length_axis1(int64_t* tomin, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t target, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_rpad_and_clip_length_axis1<uint32_t>(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset);
}
ERROR awkward_ListArray64_rpad_and_clip_length_axis1(int64_t* tomin, const int64_t* fromstarts, const int64_t* fromstops, int64_t target, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_rpad_and_clip_length_axis1<int64_t>(tomin, fromstarts, fromstops, target, lenstarts, startsoffset, stopsoffset);
}

template <typename T, typename C>
ERROR awkward_ListArray_rpad_axis1(T* toindex, const C* fromstarts, const C* fromstops, C* tostarts, C* tostops, int64_t target, int64_t length, int64_t startsoffset, int64_t stopsoffset) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    int64_t range = fromstops[startsoffset + i] - fromstarts[stopsoffset + i];
    for (int64_t j = 0; j < range; j++) {
     toindex[offset + j] = fromstarts[startsoffset + i] + j;
    }
    for (int64_t j = range; j < target; j++) {
     toindex[offset + j] = -1;
    }
    offset = (target > range) ? tostarts[i] + target : tostarts[i] + range;
    tostops[i] = offset;
   }
  return success();
}
ERROR awkward_ListArray32_rpad_axis1_64(int64_t* toindex, const int32_t* fromstarts, const int32_t* fromstops, int32_t* tostarts, int32_t* tostops, int64_t target, int64_t length, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_rpad_axis1<int64_t, int32_t>(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset);
}
ERROR awkward_ListArrayU32_rpad_axis1_64(int64_t* toindex, const uint32_t* fromstarts, const uint32_t* fromstops, uint32_t* tostarts, uint32_t* tostops, int64_t target, int64_t length, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_rpad_axis1<int64_t, uint32_t>(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset);
}
ERROR awkward_ListArray64_rpad_axis1_64(int64_t* toindex, const int64_t* fromstarts, const int64_t* fromstops, int64_t* tostarts, int64_t* tostops, int64_t target, int64_t length, int64_t startsoffset, int64_t stopsoffset) {
  return awkward_ListArray_rpad_axis1<int64_t, int64_t>(toindex, fromstarts, fromstops, tostarts, tostops, target, length, startsoffset, stopsoffset);
}

template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_and_clip_axis1(T* toindex, const C* fromoffsets, int64_t offsetsoffset, int64_t length, int64_t target) {
  for (int64_t i = 0; i < length; i++) {
    int64_t range = (T)(fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i]);
    int64_t shorter = (target < range) ? target : range;
    for (int64_t j = 0; j < shorter; j++) {
      toindex[i*target + j] = (T)fromoffsets[offsetsoffset + i] + j;
    }
    for (int64_t j = shorter; j < target; j++) {
      toindex[i*target + j] = -1;
    }
  }
  return success();
}
ERROR awkward_ListOffsetArray32_rpad_and_clip_axis1_64(int64_t* toindex, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t length, int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int32_t>(toindex, fromoffsets, offsetsoffset, length, target);
}
ERROR awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(int64_t* toindex, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t length, int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, uint32_t>(toindex, fromoffsets, offsetsoffset, length, target);
}
ERROR awkward_ListOffsetArray64_rpad_and_clip_axis1_64(int64_t* toindex, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t length, int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int64_t>(toindex, fromoffsets, offsetsoffset, length, target);
}

template <typename C>
ERROR awkward_ListOffsetArray_rpad_length_axis1(C* tooffsets, const C* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target, int64_t* tolength) {
  int64_t length = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0; i < fromlength; i++) {
    int64_t range = fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i];
    int64_t longer = (target < range) ? range : target;
    length = length + longer;
    tooffsets[i + 1] = tooffsets[i] + longer;
  }
  *tolength = length;

  return success();
}
ERROR awkward_ListOffsetArray32_rpad_length_axis1(int32_t* tooffsets, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target, int64_t* tolength) {
  return awkward_ListOffsetArray_rpad_length_axis1<int32_t>(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength);
}
ERROR awkward_ListOffsetArrayU32_rpad_length_axis1(uint32_t* tooffsets, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target, int64_t* tolength) {
  return awkward_ListOffsetArray_rpad_length_axis1<uint32_t>(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength);
}
ERROR awkward_ListOffsetArray64_rpad_length_axis1(int64_t* tooffsets, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target, int64_t* tolength) {
  return awkward_ListOffsetArray_rpad_length_axis1<int64_t>(tooffsets, fromoffsets, offsetsoffset, fromlength, target, tolength);
}

template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_axis1(T* toindex, const C* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target) {
  int64_t count = 0;
  for (int64_t i = 0; i < fromlength; i++) {
    int64_t range = (T)(fromoffsets[offsetsoffset + i + 1] - fromoffsets[offsetsoffset + i]);
    for (int64_t j = 0; j < range; j++) {
      toindex[count++] = (T)fromoffsets[offsetsoffset + i] + j;
    }
    for (int64_t j = range; j < target; j++) {
      toindex[count++] = -1;
    }
  }
  return success();
}
ERROR awkward_ListOffsetArray32_rpad_axis1_64(int64_t* toindex, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, int32_t>(toindex, fromoffsets, offsetsoffset, fromlength, target);
}
ERROR awkward_ListOffsetArrayU32_rpad_axis1_64(int64_t* toindex, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, uint32_t>(toindex, fromoffsets, offsetsoffset, fromlength, target);
}
ERROR awkward_ListOffsetArray64_rpad_axis1_64(int64_t* toindex, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t fromlength, int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, int64_t>(toindex, fromoffsets, offsetsoffset, fromlength, target);
}

template <typename T>
ERROR awkward_localindex(T* toindex, int64_t length) {
  return failure("FIXME: awkward_localindex", 0, kSliceNone);
}
ERROR awkward_localindex_64(int64_t* toindex, int64_t length) {
  return awkward_localindex<int64_t>(toindex, length);
}

template <typename C, typename T>
ERROR awkward_listarray_localindex(T* toindex, const C* offsets, int64_t offsetsoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = (int64_t)offsets[offsetsoffset + i];
    int64_t stop = (int64_t)offsets[offsetsoffset + i + 1];
    for (int64_t j = start;  j < stop;  j++) {
      toindex[j] = j - start;
    }
  }
  return success();
}
ERROR awkward_listarray32_localindex_64(int64_t* toindex, const int32_t* offsets, int64_t offsetsoffset, int64_t length) {
  return awkward_listarray_localindex<int32_t, int64_t>(toindex, offsets, offsetsoffset, length);
}
ERROR awkward_listarrayU32_localindex_64(int64_t* toindex, const uint32_t* offsets, int64_t offsetsoffset, int64_t length) {
  return awkward_listarray_localindex<uint32_t, int64_t>(toindex, offsets, offsetsoffset, length);
}
ERROR awkward_listarray64_localindex_64(int64_t* toindex, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  return awkward_listarray_localindex<int64_t, int64_t>(toindex, offsets, offsetsoffset, length);
}

template <typename T>
ERROR awkward_regulararray_localindex(T* toindex, int64_t size, int64_t length) {
  return failure("FIXME: awkward_regulararray_localindex", 0, kSliceNone);
}
ERROR awkward_regulararray_localindex_64(int64_t* toindex, int64_t size, int64_t length) {
  return awkward_regulararray_localindex<int64_t>(toindex, size, length);
}
