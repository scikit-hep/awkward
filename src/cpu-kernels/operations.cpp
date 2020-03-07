// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/operations.h"

template <typename T, typename C>
ERROR awkward_listarray_sizes(T* tosizes, const C* fromstarts, int64_t startsoffset, const C* fromstops, int64_t stopsoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[startsoffset + i];
    C stop = fromstops[stopsoffset + i];
    tosizes[i] = (T)(stop - start);
  }
  return success();
}
ERROR awkward_listarray32_sizes_64(int64_t* tosizes, const int32_t* fromstarts, int64_t startsoffset, const int32_t* fromstops, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_sizes<int64_t, int32_t>(tosizes, fromstarts, startsoffset, fromstops, stopsoffset, length);
}
ERROR awkward_listarrayU32_sizes_64(int64_t* tosizes, const uint32_t* fromstarts, int64_t startsoffset, const uint32_t* fromstops, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_sizes<int64_t, uint32_t>(tosizes, fromstarts, startsoffset, fromstops, stopsoffset, length);
}
ERROR awkward_listarray64_sizes_64(int64_t* tosizes, const int64_t* fromstarts, int64_t startsoffset, const int64_t* fromstops, int64_t stopsoffset, int64_t length) {
  return awkward_listarray_sizes<int64_t, int64_t>(tosizes, fromstarts, startsoffset, fromstops, stopsoffset, length);
}

template <typename T>
ERROR awkward_regulararray_sizes(T* tosizes, int64_t size, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    tosizes[i] = size;
  }
  return success();
}
ERROR awkward_regulararray_sizes_64(int64_t* tosizes, int64_t size, int64_t length) {
  return awkward_regulararray_sizes<int64_t>(tosizes, size, length);
}

// template <typename C>
// ERROR awkward_indexedarray_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const C* fromindex, int64_t lenindex, int64_t indexoffset) {
//   for (int64_t i = 0;  i < lenindex;  i++) {
//     C j = fromindex[indexoffset + i];
//     if (j >= lencontent) {
//       return failure("index out of range", i, j);
//     }
//     else if (j < 0) {
//       tocount[i] = 0;
//     }
//     else {
//       tocount[i] = contentcount[j];
//     }
//   }
//   return success();
// }
// ERROR awkward_indexedarray32_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const int32_t* fromindex, int64_t lenindex, int64_t indexoffset) {
//   return awkward_indexedarray_count<int32_t>(tocount, contentcount, lencontent, fromindex, lenindex, indexoffset);
// }
// ERROR awkward_indexedarrayU32_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const uint32_t* fromindex, int64_t lenindex, int64_t indexoffset) {
//   return awkward_indexedarray_count<uint32_t>(tocount, contentcount, lencontent, fromindex, lenindex, indexoffset);
// }
// ERROR awkward_indexedarray64_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const int64_t* fromindex, int64_t lenindex, int64_t indexoffset) {
//   return awkward_indexedarray_count<int64_t>(tocount, contentcount, lencontent, fromindex, lenindex, indexoffset);
// }

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
    tostarts[i] = (C)(start * scale[i]);
    tostops[i] = (C)(stop * scale[i]);
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
