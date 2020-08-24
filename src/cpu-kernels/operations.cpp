// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/operations.cpp", line)

#include "awkward/kernels/operations.h"

template <typename T, typename C>
ERROR awkward_ListArray_num(
  T* tonum,
  const C* fromstarts,
  const C* fromstops,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    tonum[i] = (T)(stop - start);
  }
  return success();
}
ERROR awkward_ListArray32_num_64(
  int64_t* tonum,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_num<int64_t, int32_t>(
    tonum,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArrayU32_num_64(
  int64_t* tonum,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_num<int64_t, uint32_t>(
    tonum,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArray64_num_64(
  int64_t* tonum,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length) {
  return awkward_ListArray_num<int64_t, int64_t>(
    tonum,
    fromstarts,
    fromstops,
    length);
}

template <typename T>
ERROR awkward_RegularArray_num(
  T* tonum,
  int64_t size,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tonum[i] = size;
  }
  return success();
}
ERROR awkward_RegularArray_num_64(
  int64_t* tonum,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_num<int64_t>(
    tonum,
    size,
    length);
}

template <typename T, typename C>
ERROR awkward_ListOffsetArray_flatten_offsets(
  T* tooffsets,
  const C* outeroffsets,
  int64_t outeroffsetslen,
  const T* inneroffsets,
  int64_t inneroffsetslen) {
  for (int64_t i = 0;  i < outeroffsetslen;  i++) {
    tooffsets[i] =
      inneroffsets[outeroffsets[i]];
  }
  return success();
}
ERROR awkward_ListOffsetArray32_flatten_offsets_64(
  int64_t* tooffsets,
  const int32_t* outeroffsets,
  int64_t outeroffsetslen,
  const int64_t* inneroffsets,
  int64_t inneroffsetslen) {
  return awkward_ListOffsetArray_flatten_offsets<int64_t, int32_t>(
    tooffsets,
    outeroffsets,
    outeroffsetslen,
    inneroffsets,
    inneroffsetslen);
}
ERROR awkward_ListOffsetArrayU32_flatten_offsets_64(
  int64_t* tooffsets,
  const uint32_t* outeroffsets,
  int64_t outeroffsetslen,
  const int64_t* inneroffsets,
  int64_t inneroffsetslen) {
  return awkward_ListOffsetArray_flatten_offsets<int64_t, uint32_t>(
    tooffsets,
    outeroffsets,
    outeroffsetslen,
    inneroffsets,
    inneroffsetslen);
}
ERROR awkward_ListOffsetArray64_flatten_offsets_64(
  int64_t* tooffsets,
  const int64_t* outeroffsets,
  int64_t outeroffsetslen,
  const int64_t* inneroffsets,
  int64_t inneroffsetslen) {
  return awkward_ListOffsetArray_flatten_offsets<int64_t, int64_t>(
    tooffsets,
    outeroffsets,
    outeroffsetslen,
    inneroffsets,
    inneroffsetslen);
}

template <typename T, typename C>
ERROR awkward_IndexedArray_flatten_none2empty(
  T* outoffsets,
  const C* outindex,
  int64_t outindexlength,
  const T* offsets,
  int64_t offsetslength) {
  outoffsets[0] = offsets[0];
  int64_t k = 1;
  for (int64_t i = 0;  i < outindexlength;  i++) {
    C idx = outindex[i];
    if (idx < 0) {
      outoffsets[k] = outoffsets[k - 1];
      k++;
    }
    else if (idx + 1 >= offsetslength) {
      return failure("flattening offset out of range", i, kSliceNone, FILENAME(__LINE__));
    }
    else {
      T count =
        offsets[idx + 1] - offsets[idx];
      outoffsets[k] = outoffsets[k - 1] + count;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_flatten_none2empty_64(
  int64_t* outoffsets,
  const int32_t* outindex,
  int64_t outindexlength,
  const int64_t* offsets,
  int64_t offsetslength) {
  return awkward_IndexedArray_flatten_none2empty<int64_t, int32_t>(
    outoffsets,
    outindex,
    outindexlength,
    offsets,
    offsetslength);
}
ERROR awkward_IndexedArrayU32_flatten_none2empty_64(
  int64_t* outoffsets,
  const uint32_t* outindex,
  int64_t outindexlength,
  const int64_t* offsets,
  int64_t offsetslength) {
  return awkward_IndexedArray_flatten_none2empty<int64_t, uint32_t>(
    outoffsets,
    outindex,
    outindexlength,
    offsets,
    offsetslength);
}
ERROR awkward_IndexedArray64_flatten_none2empty_64(
  int64_t* outoffsets,
  const int64_t* outindex,
  int64_t outindexlength,
  const int64_t* offsets,
  int64_t offsetslength) {
  return awkward_IndexedArray_flatten_none2empty<int64_t, int64_t>(
    outoffsets,
    outindex,
    outindexlength,
    offsets,
    offsetslength);
}

template <typename FROMTAGS, typename FROMINDEX, typename T>
ERROR awkward_UnionArray_flatten_length(
  int64_t* total_length,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length,
  T** offsetsraws) {
  *total_length = 0;
  for (int64_t i = 0;  i < length;  i++) {
    FROMTAGS tag = fromtags[i];
    FROMINDEX idx = fromindex[i];
    T start = offsetsraws[tag][idx];
    T stop = offsetsraws[tag][idx + 1];
    *total_length = *total_length + (stop - start);
  }
  return success();
}
ERROR awkward_UnionArray32_flatten_length_64(
  int64_t* total_length,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_length<int8_t, int32_t, int64_t>(
    total_length,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArrayU32_flatten_length_64(
  int64_t* total_length,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_length<int8_t, uint32_t, int64_t>(
    total_length,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArray64_flatten_length_64(
  int64_t* total_length,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_length<int8_t, int64_t, int64_t>(
    total_length,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}

template <typename FROMTAGS,
          typename FROMINDEX,
          typename TOTAGS,
          typename TOINDEX,
          typename T>
ERROR awkward_UnionArray_flatten_combine(
  TOTAGS* totags,
  TOINDEX* toindex,
  T* tooffsets,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t length,
  T** offsetsraws) {
  tooffsets[0] = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    FROMTAGS tag = fromtags[i];
    FROMINDEX idx = fromindex[i];
    T start = offsetsraws[tag][idx];
    T stop = offsetsraws[tag][idx + 1];
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
    for (int64_t j = start;  j < stop;  j++) {
      totags[k] = tag;
      toindex[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_UnionArray32_flatten_combine_64(
  int8_t* totags,
  int64_t* toindex,
  int64_t* tooffsets,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_combine<int8_t, int32_t, int8_t, int64_t, int64_t>(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArrayU32_flatten_combine_64(
  int8_t* totags,
  int64_t* toindex,
  int64_t* tooffsets,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_combine<int8_t, uint32_t, int8_t, int64_t, int64_t>(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}
ERROR awkward_UnionArray64_flatten_combine_64(
  int8_t* totags,
  int64_t* toindex,
  int64_t* tooffsets,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t length,
  int64_t** offsetsraws) {
  return awkward_UnionArray_flatten_combine<int8_t, int64_t, int8_t, int64_t, int64_t>(
    totags,
    toindex,
    tooffsets,
    fromtags,
    fromindex,
    length,
    offsetsraws);
}

template <typename C, typename T>
ERROR awkward_IndexedArray_flatten_nextcarry(
  T* tocarry,
  const C* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < lenindex;  i++) {
    C j = fromindex[i];
    if (j >= lencontent) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else if (j >= 0) {
      tocarry[k] = j;
      k++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_flatten_nextcarry_64(
  int64_t* tocarry,
  const int32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_flatten_nextcarry<int32_t, int64_t>(
    tocarry,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArrayU32_flatten_nextcarry_64(
  int64_t* tocarry,
  const uint32_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_flatten_nextcarry<uint32_t, int64_t>(
    tocarry,
    fromindex,
    lenindex,
    lencontent);
}
ERROR awkward_IndexedArray64_flatten_nextcarry_64(
  int64_t* tocarry,
  const int64_t* fromindex,
  int64_t lenindex,
  int64_t lencontent) {
  return awkward_IndexedArray_flatten_nextcarry<int64_t, int64_t>(
    tocarry,
    fromindex,
    lenindex,
    lencontent);
}

template <typename C, typename M, typename TO>
ERROR awkward_IndexedArray_overlay_mask(
  TO* toindex,
  const M* mask,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    M m = mask[i];
    toindex[i] = (m ? -1 : fromindex[i]);
  }
  return success();
}
ERROR awkward_IndexedArray32_overlay_mask8_to64(
  int64_t* toindex,
  const int8_t* mask,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_overlay_mask<int32_t, int8_t, int64_t>(
    toindex,
    mask,
    fromindex,
    length);
}
ERROR awkward_IndexedArrayU32_overlay_mask8_to64(
  int64_t* toindex,
  const int8_t* mask,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_overlay_mask<uint32_t, int8_t, int64_t>(
    toindex,
    mask,
    fromindex,
    length);
}
ERROR awkward_IndexedArray64_overlay_mask8_to64(
  int64_t* toindex,
  const int8_t* mask,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_overlay_mask<int64_t, int8_t, int64_t>(
    toindex,
    mask,
    fromindex,
    length);
}

template <typename C, typename M>
ERROR awkward_IndexedArray_mask(
  M* tomask,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = (fromindex[i] < 0);
  }
  return success();
}
ERROR awkward_IndexedArray32_mask8(
  int8_t* tomask,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_mask<int32_t, int8_t>(
    tomask,
    fromindex,
    length);
}
ERROR awkward_IndexedArrayU32_mask8(
  int8_t* tomask,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_mask<uint32_t, int8_t>(
    tomask,
    fromindex,
    length);
}
ERROR awkward_IndexedArray64_mask8(
  int8_t* tomask,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_IndexedArray_mask<int64_t, int8_t>(
    tomask,
    fromindex,
    length);
}

template <typename M>
ERROR awkward_ByteMaskedArray_mask(
  M* tomask,
  const M* frommask,
  int64_t length,
  bool validwhen) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = ((frommask[i] != 0) != validwhen);
  }
  return success();
}
ERROR awkward_ByteMaskedArray_mask8(
  int8_t* tomask,
  const int8_t* frommask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_mask(
    tomask,
    frommask,
    length,
    validwhen);
}

template <typename M>
ERROR awkward_zero_mask(
  M* tomask,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tomask[i] = 0;
  }
  return success();
}
ERROR awkward_zero_mask8(
  int8_t* tomask,
  int64_t length) {
  return awkward_zero_mask<int8_t>(tomask, length);
}

template <typename OUT, typename IN, typename TO>
ERROR awkward_IndexedArray_simplify(
  TO* toindex,
  const OUT* outerindex,
  int64_t outerlength,
  const IN* innerindex,
  int64_t innerlength) {
  for (int64_t i = 0;  i < outerlength;  i++) {
    OUT j = outerindex[i];
    if (j < 0) {
      toindex[i] = -1;
    }
    else if (j >= innerlength) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else {
      toindex[i] = innerindex[j];
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_simplify32_to64(
  int64_t* toindex,
  const int32_t* outerindex,
  int64_t outerlength,
  const int32_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<int32_t, int32_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArray32_simplifyU32_to64(
  int64_t* toindex,
  const int32_t* outerindex,
  int64_t outerlength,
  const uint32_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<int32_t, uint32_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArray32_simplify64_to64(
  int64_t* toindex,
  const int32_t* outerindex,
  int64_t outerlength,
  const int64_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<int32_t, int64_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArrayU32_simplify32_to64(
  int64_t* toindex,
  const uint32_t* outerindex,
  int64_t outerlength,
  const int32_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<uint32_t, int32_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArrayU32_simplifyU32_to64(
  int64_t* toindex,
  const uint32_t* outerindex,
  int64_t outerlength,
  const uint32_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<uint32_t, uint32_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArrayU32_simplify64_to64(
  int64_t* toindex,
  const uint32_t* outerindex,
  int64_t outerlength,
  const int64_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<uint32_t, int64_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArray64_simplify32_to64(
  int64_t* toindex,
  const int64_t* outerindex,
  int64_t outerlength,
  const int32_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<int64_t, int32_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArray64_simplifyU32_to64(
  int64_t* toindex,
  const int64_t* outerindex,
  int64_t outerlength,
  const uint32_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<int64_t, uint32_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}
ERROR awkward_IndexedArray64_simplify64_to64(
  int64_t* toindex,
  const int64_t* outerindex,
  int64_t outerlength,
  const int64_t* innerindex,
  int64_t innerlength) {
  return awkward_IndexedArray_simplify<int64_t, int64_t, int64_t>(
    toindex,
    outerindex,
    outerlength,
    innerindex,
    innerlength);
}

template <typename T>
ERROR awkward_RegularArray_compact_offsets(
  T* tooffsets,
  int64_t length,
  int64_t size) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[i + 1] = (i + 1)*size;
  }
  return success();
}
ERROR awkward_RegularArray_compact_offsets64(
  int64_t* tooffsets,
  int64_t length,
  int64_t size) {
  return awkward_RegularArray_compact_offsets<int64_t>(
    tooffsets,
    length,
    size);
}

template <typename C, typename T>
ERROR awkward_ListArray_compact_offsets(
  T* tooffsets,
  const C* fromstarts,
  const C* fromstops,
  int64_t length) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[i];
    C stop = fromstops[i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
  }
  return success();
}
ERROR awkward_ListArray32_compact_offsets_64(
  int64_t* tooffsets,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_compact_offsets<int32_t, int64_t>(
    tooffsets,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArrayU32_compact_offsets_64(
  int64_t* tooffsets,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t length) {
  return awkward_ListArray_compact_offsets<uint32_t, int64_t>(
    tooffsets,
    fromstarts,
    fromstops,
    length);
}
ERROR awkward_ListArray64_compact_offsets_64(
  int64_t* tooffsets,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length) {
  return awkward_ListArray_compact_offsets<int64_t, int64_t>(
    tooffsets,
    fromstarts,
    fromstops,
    length);
}

template <typename C, typename T>
ERROR awkward_ListOffsetArray_compact_offsets(
  T* tooffsets,
  const C* fromoffsets,
  int64_t length) {
  int64_t diff = (int64_t)fromoffsets[0];
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    tooffsets[i + 1] = fromoffsets[i + 1] - diff;
  }
  return success();
}
ERROR awkward_ListOffsetArray32_compact_offsets_64(
  int64_t* tooffsets,
  const int32_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_compact_offsets<int32_t, int64_t>(
    tooffsets,
    fromoffsets,
    length);
}
ERROR awkward_ListOffsetArrayU32_compact_offsets_64(
  int64_t* tooffsets,
  const uint32_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_compact_offsets<uint32_t, int64_t>(
    tooffsets,
    fromoffsets,
    length);
}
ERROR awkward_ListOffsetArray64_compact_offsets_64(
  int64_t* tooffsets,
  const int64_t* fromoffsets,
  int64_t length) {
  return awkward_ListOffsetArray_compact_offsets<int64_t, int64_t>(
    tooffsets,
    fromoffsets,
    length);
}

template <typename C, typename T>
ERROR awkward_ListArray_broadcast_tooffsets(
  T* tocarry,
  const T* fromoffsets,
  int64_t offsetslength,
  const C* fromstarts,
  const C* fromstops,
  int64_t lencontent) {
  int64_t k = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t start = (int64_t)fromstarts[i];
    int64_t stop = (int64_t)fromstops[i];
    if (start != stop  &&  stop > lencontent) {
      return failure("stops[i] > len(content)", i, stop, FILENAME(__LINE__));
    }
    int64_t count = (int64_t)(fromoffsets[i + 1] - fromoffsets[i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    if (stop - start != count) {
      return failure("cannot broadcast nested list", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = start;  j < stop;  j++) {
      tocarry[k] = (T)j;
      k++;
    }
  }
  return success();
}
ERROR awkward_ListArray32_broadcast_tooffsets_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lencontent) {
  return awkward_ListArray_broadcast_tooffsets<int32_t, int64_t>(
    tocarry,
    fromoffsets,
    offsetslength,
    fromstarts,
    fromstops,
    lencontent);
}
ERROR awkward_ListArrayU32_broadcast_tooffsets_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lencontent) {
  return awkward_ListArray_broadcast_tooffsets<uint32_t, int64_t>(
    tocarry,
    fromoffsets,
    offsetslength,
    fromstarts,
    fromstops,
    lencontent);
}
ERROR awkward_ListArray64_broadcast_tooffsets_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lencontent) {
  return awkward_ListArray_broadcast_tooffsets<int64_t, int64_t>(
    tocarry,
    fromoffsets,
    offsetslength,
    fromstarts,
    fromstops,
    lencontent);
}

template <typename T>
ERROR awkward_RegularArray_broadcast_tooffsets(
  const T* fromoffsets,
  int64_t offsetslength,
  int64_t size) {
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)(fromoffsets[i + 1] - fromoffsets[i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    if (size != count) {
      return failure("cannot broadcast nested list", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_RegularArray_broadcast_tooffsets_64(
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t size) {
  return awkward_RegularArray_broadcast_tooffsets<int64_t>(
    fromoffsets,
    offsetslength,
    size);
}

template <typename T>
ERROR awkward_RegularArray_broadcast_tooffsets_size1(
  T* tocarry,
  const T* fromoffsets,
  int64_t offsetslength) {
  int64_t k = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)(fromoffsets[i + 1] - fromoffsets[i]);
    if (count < 0) {
      return failure("broadcast's offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = 0;  j < count;  j++) {
      tocarry[k] = (T)i;
      k++;
    }
  }
  return success();
}
ERROR awkward_RegularArray_broadcast_tooffsets_size1_64(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_RegularArray_broadcast_tooffsets_size1<int64_t>(
    tocarry,
    fromoffsets,
    offsetslength);
}

template <typename C>
ERROR awkward_ListOffsetArray_toRegularArray(
  int64_t* size,
  const C* fromoffsets,
  int64_t offsetslength) {
  *size = -1;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    int64_t count = (int64_t)fromoffsets[i + 1] - (int64_t)fromoffsets[i];
    if (count < 0) {
      return failure("offsets must be monotonically increasing", i, kSliceNone, FILENAME(__LINE__));
    }
    if (*size == -1) {
      *size = count;
    }
    else if (*size != count) {
      return failure("cannot convert to RegularArray because subarray lengths are not " "regular", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  if (*size == -1) {
    *size = 0;
  }
  return success();
}
ERROR awkward_ListOffsetArray32_toRegularArray(
  int64_t* size,
  const int32_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_ListOffsetArray_toRegularArray<int32_t>(
    size,
    fromoffsets,
    offsetslength);
}
ERROR awkward_ListOffsetArrayU32_toRegularArray(
  int64_t* size,
  const uint32_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_ListOffsetArray_toRegularArray<uint32_t>(
    size,
    fromoffsets,
    offsetslength);
}
ERROR awkward_ListOffsetArray64_toRegularArray(
  int64_t* size,
  const int64_t* fromoffsets,
  int64_t offsetslength) {
  return awkward_ListOffsetArray_toRegularArray<int64_t>(
    size,
    fromoffsets,
    offsetslength);
}

template <typename TO>
ERROR
awkward_NumpyArray_fill_frombool(TO* toptr,
                                 int64_t tooffset,
                                 const bool* fromptr,
                                 int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = (TO)(fromptr[i] ? 1 : 0);
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_tobool_frombool(bool* toptr,
                                        int64_t tooffset,
                                        const bool* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill_frombool<bool>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint8_frombool(int8_t* toptr,
                                        int64_t tooffset,
                                        const bool* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill_frombool<int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_frombool(int16_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_frombool(int32_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_frombool(int64_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_frombool(uint8_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_frombool(uint16_t* toptr,
                                          int64_t tooffset,
                                          const bool* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_frombool(uint32_t* toptr,
                                          int64_t tooffset,
                                          const bool* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_frombool(uint64_t* toptr,
                                          int64_t tooffset,
                                          const bool* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_frombool(float* toptr,
                                           int64_t tooffset,
                                           const bool* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_frombool<float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_frombool(double* toptr,
                                           int64_t tooffset,
                                           const bool* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_frombool<double>(
      toptr, tooffset, fromptr, length);
}

template <typename FROM>
ERROR
awkward_NumpyArray_fill_tobool(bool* toptr,
                               int64_t tooffset,
                               const FROM* fromptr,
                               int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = fromptr[i] > 0 ? true : false;
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_tobool_fromint8(bool* toptr,
                                        int64_t tooffset,
                                        const int8_t* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill_tobool<int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromint16(bool* toptr,
                                         int64_t tooffset,
                                         const int16_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromint32(bool* toptr,
                                         int64_t tooffset,
                                         const int32_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromint64(bool* toptr,
                                         int64_t tooffset,
                                         const int64_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint8(bool* toptr,
                                         int64_t tooffset,
                                         const uint8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint16(bool* toptr,
                                          int64_t tooffset,
                                          const uint16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint32(bool* toptr,
                                          int64_t tooffset,
                                          const uint32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint64(bool* toptr,
                                          int64_t tooffset,
                                          const uint64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromfloat32(bool* toptr,
                                           int64_t tooffset,
                                           const float* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_tobool<float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromfloat64(bool* toptr,
                                           int64_t tooffset,
                                           const double* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_tobool<double>(
      toptr, tooffset, fromptr, length);
}

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill(TO* toptr,
                        int64_t tooffset,
                        const FROM* fromptr,
                        int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = (TO)fromptr[i];
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_toint8_fromint8(int8_t* toptr,
                                        int64_t tooffset,
                                        const int8_t* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint8(int16_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint8(int32_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint8(int64_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint8(uint8_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint8(uint16_t* toptr,
                                          int64_t tooffset,
                                          const int8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint8(uint32_t* toptr,
                                          int64_t tooffset,
                                          const int8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint8(uint64_t* toptr,
                                          int64_t tooffset,
                                          const int8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint8(float* toptr,
                                           int64_t tooffset,
                                           const int8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int8_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint8(double* toptr,
                                           int64_t tooffset,
                                           const int8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int8_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromint16(int8_t* toptr,
                                         int64_t tooffset,
                                         const int16_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint16(int16_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint16(int32_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint16(int64_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint16(uint8_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint16(uint16_t* toptr,
                                           int64_t tooffset,
                                           const int16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint16(uint32_t* toptr,
                                           int64_t tooffset,
                                           const int16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint16(uint64_t* toptr,
                                           int64_t tooffset,
                                           const int16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint16(float* toptr,
                                            int64_t tooffset,
                                            const int16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int16_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint16(double* toptr,
                                            int64_t tooffset,
                                            const int16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int16_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromint32(int8_t* toptr,
                                         int64_t tooffset,
                                         const int32_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint32(int16_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint32(int32_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint32(int64_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint32(uint8_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint32(uint16_t* toptr,
                                           int64_t tooffset,
                                           const int32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint32(uint32_t* toptr,
                                           int64_t tooffset,
                                           const int32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint32(uint64_t* toptr,
                                           int64_t tooffset,
                                           const int32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint32(float* toptr,
                                            int64_t tooffset,
                                            const int32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int32_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint32(double* toptr,
                                            int64_t tooffset,
                                            const int32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int32_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromint64(int8_t* toptr,
                                         int64_t tooffset,
                                         const int64_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint64(int16_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint64(int32_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint64(int64_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint64(uint8_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint64(uint16_t* toptr,
                                           int64_t tooffset,
                                           const int64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint64(uint32_t* toptr,
                                           int64_t tooffset,
                                           const int64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint64(uint64_t* toptr,
                                           int64_t tooffset,
                                           const int64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint64(float* toptr,
                                            int64_t tooffset,
                                            const int64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int64_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint64(double* toptr,
                                            int64_t tooffset,
                                            const int64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int64_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint8(int8_t* toptr,
                                         int64_t tooffset,
                                         const uint8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint8(int16_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint8(int32_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint8(int64_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint8(uint8_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint8(uint16_t* toptr,
                                           int64_t tooffset,
                                           const uint8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint8(uint32_t* toptr,
                                           int64_t tooffset,
                                           const uint8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint8(uint64_t* toptr,
                                           int64_t tooffset,
                                           const uint8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint8(float* toptr,
                                            int64_t tooffset,
                                            const uint8_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint8(double* toptr,
                                            int64_t tooffset,
                                            const uint8_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint16(int8_t* toptr,
                                          int64_t tooffset,
                                          const uint16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint16(int16_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint16(int32_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint16(int64_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint16(uint8_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint16(uint16_t* toptr,
                                            int64_t tooffset,
                                            const uint16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint16(uint32_t* toptr,
                                            int64_t tooffset,
                                            const uint16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint16(uint64_t* toptr,
                                            int64_t tooffset,
                                            const uint16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint16(float* toptr,
                                             int64_t tooffset,
                                             const uint16_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint16(double* toptr,
                                             int64_t tooffset,
                                             const uint16_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint32(int8_t* toptr,
                                          int64_t tooffset,
                                          const uint32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint32(int16_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint32(int32_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint32(int64_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint32(uint8_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint32(uint16_t* toptr,
                                            int64_t tooffset,
                                            const uint32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint32(uint32_t* toptr,
                                            int64_t tooffset,
                                            const uint32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint32(uint64_t* toptr,
                                            int64_t tooffset,
                                            const uint32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint32(float* toptr,
                                             int64_t tooffset,
                                             const uint32_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint32(double* toptr,
                                             int64_t tooffset,
                                             const uint32_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint64(int8_t* toptr,
                                          int64_t tooffset,
                                          const uint64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint64(int16_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint64(int32_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint64(int64_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint64(uint8_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint64(uint16_t* toptr,
                                            int64_t tooffset,
                                            const uint64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint64(uint32_t* toptr,
                                            int64_t tooffset,
                                            const uint64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint64(uint64_t* toptr,
                                            int64_t tooffset,
                                            const uint64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint64(float* toptr,
                                             int64_t tooffset,
                                             const uint64_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint64(double* toptr,
                                             int64_t tooffset,
                                             const uint64_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromfloat32(int8_t* toptr,
                                           int64_t tooffset,
                                           const float* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<float, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromfloat32(int16_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromfloat32(int32_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromfloat32(int64_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromfloat32(uint8_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromfloat32(uint16_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<float, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromfloat32(uint32_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<float, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromfloat32(uint64_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<float, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromfloat32(float* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<float, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromfloat32(double* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<float, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromfloat64(int8_t* toptr,
                                           int64_t tooffset,
                                           const double* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<double, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromfloat64(int16_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromfloat64(int32_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromfloat64(int64_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromfloat64(uint8_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromfloat64(uint16_t* toptr,
                                             int64_t tooffset,
                                             const double* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<double, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromfloat64(uint32_t* toptr,
                                             int64_t tooffset,
                                             const double* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<double, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromfloat64(uint64_t* toptr,
                                             int64_t tooffset,
                                             const double* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<double, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromfloat64(float* toptr,
                                              int64_t tooffset,
                                              const double* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<double, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromfloat64(double* toptr,
                                              int64_t tooffset,
                                              const double* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<double, double>(
      toptr, tooffset, fromptr, length);
}

template <typename FROM, typename TO>
ERROR awkward_ListArray_fill(
  TO* tostarts,
  int64_t tostartsoffset,
  TO* tostops,
  int64_t tostopsoffset,
  const FROM* fromstarts,
  const FROM* fromstops,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    tostarts[tostartsoffset + i] = (TO)(fromstarts[i] + base);
    tostops[tostopsoffset + i] = (TO)(fromstops[i] + base);
  }
  return success();
}
ERROR awkward_ListArray_fill_to64_from32(
  int64_t* tostarts,
  int64_t tostartsoffset,
  int64_t* tostops,
  int64_t tostopsoffset,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t length,
  int64_t base) {
  return awkward_ListArray_fill<int32_t, int64_t>(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base);
}
ERROR awkward_ListArray_fill_to64_fromU32(
  int64_t* tostarts,
  int64_t tostartsoffset,
  int64_t* tostops,
  int64_t tostopsoffset,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t length,
  int64_t base) {
  return awkward_ListArray_fill<uint32_t, int64_t>(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base);
}
ERROR awkward_ListArray_fill_to64_from64(
  int64_t* tostarts,
  int64_t tostartsoffset,
  int64_t* tostops,
  int64_t tostopsoffset,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t base) {
  return awkward_ListArray_fill<int64_t, int64_t>(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base);
}

template <typename FROM, typename TO>
ERROR awkward_IndexedArray_fill(
  TO* toindex,
  int64_t toindexoffset,
  const FROM* fromindex,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    FROM fromval = fromindex[i];
    toindex[toindexoffset + i] = fromval < 0 ? -1 : (TO)(fromval + base);
  }
  return success();
}
ERROR awkward_IndexedArray_fill_to64_from32(
  int64_t* toindex,
  int64_t toindexoffset,
  const int32_t* fromindex,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill<int32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length,
    base);
}
ERROR awkward_IndexedArray_fill_to64_fromU32(
  int64_t* toindex,
  int64_t toindexoffset,
  const uint32_t* fromindex,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill<uint32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length,
    base);
}
ERROR awkward_IndexedArray_fill_to64_from64(
  int64_t* toindex,
  int64_t toindexoffset,
  const int64_t* fromindex,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill<int64_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length,
    base);
}

template <typename TO>
ERROR awkward_IndexedArray_fill_count(
  TO* toindex,
  int64_t toindexoffset,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = i + base;
  }
  return success();
}
ERROR awkward_IndexedArray_fill_to64_count(
  int64_t* toindex,
  int64_t toindexoffset,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill_count(
    toindex,
    toindexoffset,
    length,
    base);
}

template <typename FROM, typename TO>
ERROR awkward_UnionArray_filltags(
  TO* totags,
  int64_t totagsoffset,
  const FROM* fromtags,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    totags[totagsoffset + i] = (TO)(fromtags[i] + base);
  }
  return success();
}
ERROR awkward_UnionArray_filltags_to8_from8(
  int8_t* totags,
  int64_t totagsoffset,
  const int8_t* fromtags,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_filltags<int8_t, int8_t>(
    totags,
    totagsoffset,
    fromtags,
    length,
    base);
}

template <typename FROM, typename TO>
ERROR awkward_UnionArray_fillindex(
  TO* toindex,
  int64_t toindexoffset,
  const FROM* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = (TO)fromindex[i];
  }
  return success();
}
ERROR awkward_UnionArray_fillindex_to64_from32(
  int64_t* toindex,
  int64_t toindexoffset,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillindex<int32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillindex_to64_fromU32(
  int64_t* toindex,
  int64_t toindexoffset,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillindex<uint32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillindex_to64_from64(
  int64_t* toindex,
  int64_t toindexoffset,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillindex<int64_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length);
}

template <typename TO>
ERROR awkward_UnionArray_filltags_const(
  TO* totags,
  int64_t totagsoffset,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    totags[totagsoffset + i] = (TO)base;
  }
  return success();
}
ERROR awkward_UnionArray_filltags_to8_const(
  int8_t* totags,
  int64_t totagsoffset,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_filltags_const<int8_t>(
    totags,
    totagsoffset,
    length,
    base);
}

template <typename TO>
ERROR awkward_UnionArray_fillindex_count(
  TO* toindex,
  int64_t toindexoffset,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = (TO)i;
  }
  return success();
}
ERROR awkward_UnionArray_fillindex_to64_count(
  int64_t* toindex,
  int64_t toindexoffset,
  int64_t length) {
  return awkward_UnionArray_fillindex_count<int64_t>(
    toindex,
    toindexoffset,
    length);
}

template <typename OUTERTAGS,
          typename OUTERINDEX,
          typename INNERTAGS,
          typename INNERINDEX,
          typename TOTAGS,
          typename TOINDEX>
ERROR awkward_UnionArray_simplify(
  TOTAGS* totags,
  TOINDEX* toindex,
  const OUTERTAGS* outertags,
  const OUTERINDEX* outerindex,
  const INNERTAGS* innertags,
  const INNERINDEX* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    if (outertags[i] == outerwhich) {
      OUTERINDEX j = outerindex[i];
      if (innertags[j] == innerwhich) {
        totags[i] = (TOTAGS)towhich;
        toindex[i] = (TOINDEX)(innerindex[j] + base);
      }
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_simplify8_32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int32_t* outerindex,
  const int8_t* innertags,
  const int32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int32_t, int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_32_simplify8_U32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int32_t* outerindex,
  const int8_t* innertags,
  const uint32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int32_t, int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_32_simplify8_64_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int32_t* outerindex,
  const int8_t* innertags,
  const int64_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int32_t, int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify8_32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const uint32_t* outerindex,
  const int8_t* innertags,
  const int32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, uint32_t, int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify8_U32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const uint32_t* outerindex,
  const int8_t* innertags,
  const uint32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, uint32_t, int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify8_64_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const uint32_t* outerindex,
  const int8_t* innertags,
  const int64_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, uint32_t, int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify8_32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int64_t* outerindex,
  const int8_t* innertags,
  const int32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int64_t, int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify8_U32_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int64_t* outerindex,
  const int8_t* innertags,
  const uint32_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int64_t, int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify8_64_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* outertags,
  const int64_t* outerindex,
  const int8_t* innertags,
  const int64_t* innerindex,
  int64_t towhich,
  int64_t innerwhich,
  int64_t outerwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify<int8_t, int64_t, int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base);
}

template <typename FROMTAGS,
          typename FROMINDEX,
          typename TOTAGS,
          typename TOINDEX>
ERROR awkward_UnionArray_simplify_one(
  TOTAGS* totags,
  TOINDEX* toindex,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    if (fromtags[i] == fromwhich) {
      totags[i] = (TOTAGS)towhich;
      toindex[i] = (TOINDEX)(fromindex[i] + base);
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_simplify_one_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify_one<int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    fromtags,
    fromindex,
    towhich,
    fromwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify_one_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify_one<int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    fromtags,
    fromindex,
    towhich,
    fromwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify_one_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify_one<int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    fromtags,
    fromindex,
    towhich,
    fromwhich,
    length,
    base);
}

template <typename C>
ERROR awkward_ListArray_validity(
  const C* starts,
  const C* stops,
  int64_t length,
  int64_t lencontent) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = starts[i];
    C stop = stops[i];
    if (start != stop) {
      if (start > stop) {
        return failure("start[i] > stop[i]", i, kSliceNone, FILENAME(__LINE__));
      }
      if (start < 0) {
        return failure("start[i] < 0", i, kSliceNone, FILENAME(__LINE__));
      }
      if (stop > lencontent) {
        return failure("stop[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
      }
    }
  }
  return success();
}

ERROR awkward_ListArray32_validity(
  const int32_t* starts,
  const int32_t* stops,
  int64_t length,
  int64_t lencontent) {
  return awkward_ListArray_validity<int32_t>(
    starts,
    stops,
    length,
    lencontent);
}
ERROR awkward_ListArrayU32_validity(
  const uint32_t* starts,
  const uint32_t* stops,
  int64_t length,
  int64_t lencontent) {
  return awkward_ListArray_validity<uint32_t>(
    starts,
    stops,
    length,
    lencontent);
}
ERROR awkward_ListArray64_validity(
  const int64_t* starts,
  const int64_t* stops,
  int64_t length,
  int64_t lencontent) {
  return awkward_ListArray_validity<int64_t>(
    starts,
    stops,
    length,
    lencontent);
}

template <typename C>
ERROR awkward_IndexedArray_validity(
  const C* index,
  int64_t length,
  int64_t lencontent,
  bool isoption) {
  for (int64_t i = 0;  i < length;  i++) {
    C idx = index[i];
    if (!isoption) {
      if (idx < 0) {
        return failure("index[i] < 0", i, kSliceNone, FILENAME(__LINE__));
      }
    }
    if (idx >= lencontent) {
      return failure("index[i] >= len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_validity(
  const int32_t* index,
  int64_t length,
  int64_t lencontent,
  bool isoption) {
  return awkward_IndexedArray_validity<int32_t>(
    index,
    length,
    lencontent,
    isoption);
}
ERROR awkward_IndexedArrayU32_validity(
  const uint32_t* index,
  int64_t length,
  int64_t lencontent,
  bool isoption) {
  return awkward_IndexedArray_validity<uint32_t>(
    index,
    length,
    lencontent,
    isoption);
}
ERROR awkward_IndexedArray64_validity(
  const int64_t* index,
  int64_t length,
  int64_t lencontent,
  bool isoption) {
  return awkward_IndexedArray_validity<int64_t>(
    index,
    length,
    lencontent,
    isoption);
}

template <typename T, typename I>
ERROR awkward_UnionArray_validity(
  const T* tags,
  const I* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  for (int64_t i = 0;  i < length;  i++) {
    T tag = tags[i];
    I idx = index[i];
    if (tag < 0) {
      return failure("tags[i] < 0", i, kSliceNone, FILENAME(__LINE__));
    }
    if (idx < 0) {
      return failure("index[i] < 0", i, kSliceNone, FILENAME(__LINE__));
    }
    if (tag >= numcontents) {
      return failure("tags[i] >= len(contents)", i, kSliceNone, FILENAME(__LINE__));
    }
    int64_t lencontent = lencontents[tag];
    if (idx >= lencontent) {
      return failure("index[i] >= len(content[tags[i]])", i, kSliceNone, FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_validity(
  const int8_t* tags,
  const int32_t* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  return awkward_UnionArray_validity<int8_t, int32_t>(
    tags,
    index,
    length,
    numcontents,
    lencontents);
}
ERROR awkward_UnionArray8_U32_validity(
  const int8_t* tags,
  const uint32_t* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  return awkward_UnionArray_validity<int8_t, uint32_t>(
    tags,
    index,
    length,
    numcontents,
    lencontents);
}
ERROR awkward_UnionArray8_64_validity(
  const int8_t* tags,
  const int64_t* index,
  int64_t length,
  int64_t numcontents,
  const int64_t* lencontents) {
  return awkward_UnionArray_validity<int8_t, int64_t>(
    tags,
    index,
    length,
    numcontents,
    lencontents);
}

template <typename T, typename C>
ERROR awkward_UnionArray_fillna(
  T* toindex,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0; i < length; i++)
  {
    toindex[i] = fromindex[i] >= 0 ? fromindex[i] : 0;
  }
  return success();
}
ERROR awkward_UnionArray_fillna_from32_to64(
  int64_t* toindex,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillna<int64_t, int32_t>(
    toindex,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillna_fromU32_to64(
  int64_t* toindex,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillna<int64_t, uint32_t>(
    toindex,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillna_from64_to64(
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillna<int64_t, int64_t>(
    toindex,
    fromindex,
    length);
}

template <typename T>
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1(
  T* toindex,
  const int8_t* frommask,
  int64_t length) {
  int64_t count = 0;
  for (int64_t i = 0; i < length; i++) {
    if (frommask[i]) {
      toindex[i] = -1;
    }
    else {
      toindex[i] = count;
      count++;
    }
  }
  return success();
}
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
  int64_t* toindex,
  const int8_t* frommask,
  int64_t length) {
  return awkward_IndexedOptionArray_rpad_and_clip_mask_axis1<int64_t>(
    toindex,
    frommask,
    length);
}

template <typename T>
ERROR awkward_index_rpad_and_clip_axis0(
  T* toindex,
  int64_t target,
  int64_t length) {
  int64_t shorter = (target < length ? target : length);
  for (int64_t i = 0; i < shorter; i++) {
    toindex[i] = i;
  }
  for (int64_t i = shorter; i < target; i++) {
    toindex[i] = -1;
  }
  return success();
}
ERROR awkward_index_rpad_and_clip_axis0_64(
  int64_t* toindex,
  int64_t target,
  int64_t length) {
  return awkward_index_rpad_and_clip_axis0<int64_t>(
    toindex,
    target,
    length);
}

template <typename T>
ERROR awkward_index_rpad_and_clip_axis1(
  T* tostarts,
  T* tostops,
  int64_t target,
  int64_t length) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    offset = offset + target;
    tostops[i] = offset;
   }
  return success();
}
ERROR awkward_index_rpad_and_clip_axis1_64(
  int64_t* tostarts,
  int64_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_index_rpad_and_clip_axis1<int64_t>(
    tostarts,
    tostops,
    target,
    length);
}

template <typename T>
ERROR awkward_RegularArray_rpad_and_clip_axis1(
  T* toindex,
  int64_t target,
  int64_t size,
  int64_t length) {
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
ERROR awkward_RegularArray_rpad_and_clip_axis1_64(
  int64_t* toindex,
  int64_t target,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_rpad_and_clip_axis1<int64_t>(
    toindex,
    target,
    size,
    length);
}

template <typename C>
ERROR awkward_ListArray_min_range(
  int64_t* tomin,
  const C* fromstarts,
  const C* fromstops,
  int64_t lenstarts) {
  int64_t shorter = fromstops[0] - fromstarts[0];
  for (int64_t i = 1;  i < lenstarts;  i++) {
    int64_t rangeval = fromstops[i] - fromstarts[i];
    shorter = (shorter < rangeval) ? shorter : rangeval;
  }
  *tomin = shorter;
  return success();
}
ERROR awkward_ListArray32_min_range(
  int64_t* tomin,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t lenstarts) {
  return awkward_ListArray_min_range<int32_t>(
    tomin,
    fromstarts,
    fromstops,
    lenstarts);
}
ERROR awkward_ListArrayU32_min_range(
  int64_t* tomin,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t lenstarts) {
  return awkward_ListArray_min_range<uint32_t>(
    tomin,
    fromstarts,
    fromstops,
    lenstarts);
}
ERROR awkward_ListArray64_min_range(
  int64_t* tomin,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t lenstarts) {
  return awkward_ListArray_min_range<int64_t>(
    tomin,
    fromstarts,
    fromstops,
    lenstarts);
}

template <typename C>
ERROR awkward_ListArray_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const C* fromstarts,
  const C* fromstops,
  int64_t target,
  int64_t lenstarts) {
  int64_t length = 0;
  for (int64_t i = 0;  i < lenstarts;  i++) {
    int64_t rangeval = fromstops[i] - fromstarts[i];
    length += (target > rangeval) ? target : rangeval;
  }
  *tomin = length;
  return success();
}
ERROR awkward_ListArray32_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t target,
  int64_t lenstarts) {
  return awkward_ListArray_rpad_and_clip_length_axis1<int32_t>(
    tomin,
    fromstarts,
    fromstops,
    target,
    lenstarts);
}
ERROR awkward_ListArrayU32_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t target,
  int64_t lenstarts) {
  return awkward_ListArray_rpad_and_clip_length_axis1<uint32_t>(
    tomin,
    fromstarts,
    fromstops,
    target,
    lenstarts);
}
ERROR awkward_ListArray64_rpad_and_clip_length_axis1(
  int64_t* tomin,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t target,
  int64_t lenstarts) {
  return awkward_ListArray_rpad_and_clip_length_axis1<int64_t>(
    tomin,
    fromstarts,
    fromstops,
    target,
    lenstarts);
}

template <typename T, typename C>
ERROR awkward_ListArray_rpad_axis1(
  T* toindex,
  const C* fromstarts,
  const C* fromstops,
  C* tostarts,
  C* tostops,
  int64_t target,
  int64_t length) {
  int64_t offset = 0;
  for (int64_t i = 0; i < length; i++) {
    tostarts[i] = offset;
    int64_t rangeval = fromstops[i] - fromstarts[i];
    for (int64_t j = 0; j < rangeval; j++) {
     toindex[offset + j] = fromstarts[i] + j;
    }
    for (int64_t j = rangeval; j < target; j++) {
     toindex[offset + j] = -1;
    }
    offset = (target > rangeval) ? tostarts[i] + target : tostarts[i] + rangeval;
    tostops[i] = offset;
   }
  return success();
}
ERROR awkward_ListArray32_rpad_axis1_64(
  int64_t* toindex,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int32_t* tostarts,
  int32_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_ListArray_rpad_axis1<int64_t, int32_t>(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length);
}
ERROR awkward_ListArrayU32_rpad_axis1_64(
  int64_t* toindex,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  uint32_t* tostarts,
  uint32_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_ListArray_rpad_axis1<int64_t, uint32_t>(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length);
}
ERROR awkward_ListArray64_rpad_axis1_64(
  int64_t* toindex,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t* tostarts,
  int64_t* tostops,
  int64_t target,
  int64_t length) {
  return awkward_ListArray_rpad_axis1<int64_t, int64_t>(
    toindex,
    fromstarts,
    fromstops,
    tostarts,
    tostops,
    target,
    length);
}

template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_and_clip_axis1(
  T* toindex,
  const C* fromoffsets,
  int64_t length,
  int64_t target) {
  for (int64_t i = 0; i < length; i++) {
    int64_t rangeval = (T)(fromoffsets[i + 1] - fromoffsets[i]);
    int64_t shorter = (target < rangeval) ? target : rangeval;
    for (int64_t j = 0; j < shorter; j++) {
      toindex[i*target + j] = (T)fromoffsets[i] + j;
    }
    for (int64_t j = shorter; j < target; j++) {
      toindex[i*target + j] = -1;
    }
  }
  return success();
}
ERROR awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
  int64_t* toindex,
  const int32_t* fromoffsets,
  int64_t length,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int32_t>(
    toindex,
    fromoffsets,
    length,
    target);
}
ERROR awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
  int64_t* toindex,
  const uint32_t* fromoffsets,
  int64_t length,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, uint32_t>(
    toindex,
    fromoffsets,
    length,
    target);
}
ERROR awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
  int64_t* toindex,
  const int64_t* fromoffsets,
  int64_t length,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int64_t>(
    toindex,
    fromoffsets,
    length,
    target);
}

template <typename C>
ERROR awkward_ListOffsetArray_rpad_length_axis1(
  C* tooffsets,
  const C* fromoffsets,
  int64_t fromlength,
  int64_t target,
  int64_t* tolength) {
  int64_t length = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0; i < fromlength; i++) {
    int64_t rangeval =
      fromoffsets[i + 1] - fromoffsets[i];
    int64_t longer = (target < rangeval) ? rangeval : target;
    length = length + longer;
    tooffsets[i + 1] = tooffsets[i] + longer;
  }
  *tolength = length;

  return success();
}
ERROR awkward_ListOffsetArray32_rpad_length_axis1(
  int32_t* tooffsets,
  const int32_t* fromoffsets,
  int64_t fromlength,
  int64_t target,
  int64_t* tolength) {
  return awkward_ListOffsetArray_rpad_length_axis1<int32_t>(
    tooffsets,
    fromoffsets,
    fromlength,
    target,
    tolength);
}
ERROR awkward_ListOffsetArrayU32_rpad_length_axis1(
  uint32_t* tooffsets,
  const uint32_t* fromoffsets,
  int64_t fromlength,
  int64_t target,
  int64_t* tolength) {
  return awkward_ListOffsetArray_rpad_length_axis1<uint32_t>(
    tooffsets,
    fromoffsets,
    fromlength,
    target,
    tolength);
}
ERROR awkward_ListOffsetArray64_rpad_length_axis1(
  int64_t* tooffsets,
  const int64_t* fromoffsets,
  int64_t fromlength,
  int64_t target,
  int64_t* tolength) {
  return awkward_ListOffsetArray_rpad_length_axis1<int64_t>(
    tooffsets,
    fromoffsets,
    fromlength,
    target,
    tolength);
}

template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_axis1(
  T* toindex,
  const C* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  int64_t count = 0;
  for (int64_t i = 0; i < fromlength; i++) {
    int64_t rangeval =
      (T)(fromoffsets[i + 1] - fromoffsets[i]);
    for (int64_t j = 0; j < rangeval; j++) {
      toindex[count] = (T)fromoffsets[i] + j;
      count++;
    }
    for (int64_t j = rangeval; j < target; j++) {
      toindex[count] = -1;
      count++;
    }
  }
  return success();
}
ERROR awkward_ListOffsetArray32_rpad_axis1_64(
  int64_t* toindex,
  const int32_t* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, int32_t>(
    toindex,
    fromoffsets,
    fromlength,
    target);
}
ERROR awkward_ListOffsetArrayU32_rpad_axis1_64(
  int64_t* toindex,
  const uint32_t* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, uint32_t>(
    toindex,
    fromoffsets,
    fromlength,
    target);
}
ERROR awkward_ListOffsetArray64_rpad_axis1_64(
  int64_t* toindex,
  const int64_t* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, int64_t>(
    toindex,
    fromoffsets,
    fromlength,
    target);
}

template <typename T>
ERROR awkward_localindex(
  T* toindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[i] = i;
  }
  return success();
}
ERROR awkward_localindex_64(
  int64_t* toindex,
  int64_t length) {
  return awkward_localindex<int64_t>(
    toindex,
    length);
}

template <typename C, typename T>
ERROR awkward_ListArray_localindex(
  T* toindex,
  const C* offsets,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = (int64_t)offsets[i];
    int64_t stop = (int64_t)offsets[i + 1];
    for (int64_t j = start;  j < stop;  j++) {
      toindex[j] = j - start;
    }
  }
  return success();
}
ERROR awkward_ListArray32_localindex_64(
  int64_t* toindex,
  const int32_t* offsets,
  int64_t length) {
  return awkward_ListArray_localindex<int32_t, int64_t>(
    toindex,
    offsets,
    length);
}
ERROR awkward_ListArrayU32_localindex_64(
  int64_t* toindex,
  const uint32_t* offsets,
  int64_t length) {
  return awkward_ListArray_localindex<uint32_t, int64_t>(
    toindex,
    offsets,
    length);
}
ERROR awkward_ListArray64_localindex_64(
  int64_t* toindex,
  const int64_t* offsets,
  int64_t length) {
  return awkward_ListArray_localindex<int64_t, int64_t>(
    toindex,
    offsets,
    length);
}

template <typename T>
ERROR awkward_RegularArray_localindex(
  T* toindex,
  int64_t size,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      toindex[i*size + j] = j;
    }
  }
  return success();
}
ERROR awkward_RegularArray_localindex_64(
  int64_t* toindex,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_localindex<int64_t>(
    toindex,
    size,
    length);
}

template <typename T>
ERROR awkward_combinations(
  T* toindex,
  int64_t n,
  bool replacement,
  int64_t singlelen) {
  return failure("FIXME: awkward_combinations", 0, kSliceNone, FILENAME(__LINE__));
}
ERROR awkward_combinations_64(
  int64_t* toindex,
  int64_t n,
  bool replacement,
  int64_t singlelen) {
  return awkward_combinations<int64_t>(
    toindex,
    n,
    replacement,
    singlelen);
}

template <typename C, typename T>
ERROR awkward_ListArray_combinations_length(
  int64_t* totallen,
  T* tooffsets,
  int64_t n,
  bool replacement,
  const C* starts,
  const C* stops,
  int64_t length) {
  *totallen = 0;
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    int64_t size = (int64_t)(stops[i] - starts[i]);
    if (replacement) {
      size += (n - 1);
    }
    int64_t thisn = n;
    int64_t combinationslen;
    if (thisn > size) {
      combinationslen = 0;
    }
    else if (thisn == size) {
      combinationslen = 1;
    }
    else {
      if (thisn * 2 > size) {
        thisn = size - thisn;
      }
      combinationslen = size;
      for (int64_t j = 2;  j <= thisn;  j++) {
        combinationslen *= (size - j + 1);
        combinationslen /= j;
      }
    }
    *totallen = *totallen + combinationslen;
    tooffsets[i + 1] = tooffsets[i] + combinationslen;
  }
  return success();
}
ERROR awkward_ListArray32_combinations_length_64(
  int64_t* totallen,
  int64_t* tooffsets,
  int64_t n,
  bool replacement,
  const int32_t* starts,
  const int32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations_length<int32_t, int64_t>(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArrayU32_combinations_length_64(
  int64_t* totallen,
  int64_t* tooffsets,
  int64_t n,
  bool replacement,
  const uint32_t* starts,
  const uint32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations_length<uint32_t, int64_t>(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArray64_combinations_length_64(
  int64_t* totallen,
  int64_t* tooffsets,
  int64_t n,
  bool replacement,
  const int64_t* starts,
  const int64_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations_length<int64_t, int64_t>(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length);
}

template <typename T>
void awkward_ListArray_combinations_step(
  T** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t j,
  int64_t stop,
  int64_t n,
  bool replacement) {
  while (fromindex[j] < stop) {
    if (replacement) {
      for (int64_t k = j + 1;  k < n;  k++) {
        fromindex[k] = fromindex[j];
      }
    }
    else {
      for (int64_t k = j + 1;  k < n;  k++) {
        fromindex[k] = fromindex[j] + (k - j);
      }
    }
    if (j + 1 == n) {
      for (int64_t k = 0;  k < n;  k++) {
        tocarry[k][toindex[k]] = fromindex[k];
        toindex[k]++;
      }
    }
    else {
      awkward_ListArray_combinations_step<T>(
        tocarry,
        toindex,
        fromindex,
        j + 1,
        stop,
        n,
        replacement);
    }
    fromindex[j]++;
  }
}

template <typename C, typename T>
ERROR awkward_ListArray_combinations(
  T** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const C* starts,
  const C* stops,
  int64_t length) {
  for (int64_t j = 0;  j < n;  j++) {
    toindex[j] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = (int64_t)starts[i];
    int64_t stop = (int64_t)stops[i];
    fromindex[0] = start;
    awkward_ListArray_combinations_step<T>(
      tocarry,
      toindex,
      fromindex,
      0,
      stop,
      n,
      replacement);
  }
  return success();
}
ERROR awkward_ListArray32_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const int32_t* starts,
  const int32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations<int32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArrayU32_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const uint32_t* starts,
  const uint32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations<uint32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArray64_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const int64_t* starts,
  const int64_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations<int64_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length);
}

template <typename C, typename T>
ERROR awkward_RegularArray_combinations(
  T** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  int64_t size,
  int64_t length) {
  for (int64_t j = 0;  j < n;  j++) {
    toindex[j] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = size*i;
    int64_t stop = start + size;
    fromindex[0] = start;
    awkward_ListArray_combinations_step<T>(
      tocarry,
      toindex,
      fromindex,
      0,
      stop,
      n,
      replacement);
  }
  return success();
}
ERROR awkward_RegularArray_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_combinations<int32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    size,
    length);
}

template <typename M>
ERROR awkward_ByteMaskedArray_overlay_mask(
  M* tomask,
  const M* theirmask,
  const M* mymask,
  int64_t length,
  bool validwhen) {
  for (int64_t i = 0;  i < length;  i++) {
    bool theirs = theirmask[i];
    bool mine = ((mymask[i] != 0) != validwhen);
    tomask[i] = ((theirs | mine) ? 1 : 0);
  }
  return success();
}
ERROR awkward_ByteMaskedArray_overlay_mask8(
  int8_t* tomask,
  const int8_t* theirmask,
  const int8_t* mymask,
  int64_t length,
  bool validwhen) {
  return awkward_ByteMaskedArray_overlay_mask<int8_t>(
    tomask,
    theirmask,
    mymask,
    length,
    validwhen);
}

ERROR awkward_BitMaskedArray_to_ByteMaskedArray(
  int8_t* tobytemask,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order) {
  if (lsb_order) {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      tobytemask[i*8 + 0] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 1] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 2] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 3] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 4] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 5] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 6] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 7] = ((byte & ((uint8_t)1)) != validwhen);
    }
  }
  else {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      tobytemask[i*8 + 0] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 1] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 2] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 3] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 4] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 5] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 6] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 7] = (((byte & ((uint8_t)128)) != 0) != validwhen);
    }
  }
  return success();
}

template <typename T>
ERROR awkward_BitMaskedArray_to_IndexedOptionArray(
  T* toindex,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order) {
  if (lsb_order) {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 0] = i*8 + 0;
      }
      else {
        toindex[i*8 + 0] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 1] = i*8 + 1;
      }
      else {
        toindex[i*8 + 1] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 2] = i*8 + 2;
      }
      else {
        toindex[i*8 + 2] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 3] = i*8 + 3;
      }
      else {
        toindex[i*8 + 3] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 4] = i*8 + 4;
      }
      else {
        toindex[i*8 + 4] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 5] = i*8 + 5;
      }
      else {
        toindex[i*8 + 5] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 6] = i*8 + 6;
      }
      else {
        toindex[i*8 + 6] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 7] = i*8 + 7;
      }
      else {
        toindex[i*8 + 7] = -1;
      }
    }
  }
  else {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 0] = i*8 + 0;
      }
      else {
        toindex[i*8 + 0] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 1] = i*8 + 1;
      }
      else {
        toindex[i*8 + 1] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 2] = i*8 + 2;
      }
      else {
        toindex[i*8 + 2] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 3] = i*8 + 3;
      }
      else {
        toindex[i*8 + 3] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 4] = i*8 + 4;
      }
      else {
        toindex[i*8 + 4] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 5] = i*8 + 5;
      }
      else {
        toindex[i*8 + 5] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 6] = i*8 + 6;
      }
      else {
        toindex[i*8 + 6] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 7] = i*8 + 7;
      }
      else {
        toindex[i*8 + 7] = -1;
      }
    }
  }
  return success();
}
ERROR awkward_BitMaskedArray_to_IndexedOptionArray64(
  int64_t* toindex,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order) {
  return awkward_BitMaskedArray_to_IndexedOptionArray<int64_t>(
    toindex,
    frombitmask,
    bitmasklength,
    validwhen,
    lsb_order);
}
