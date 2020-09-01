// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/identities.cpp", line)

#include "awkward/kernels/identities.h"

template <typename T>
ERROR awkward_new_Identities(
  T* toptr,
  int64_t length) {
  for (T i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_new_Identities32(
  int32_t* toptr,
  int64_t length) {
  return awkward_new_Identities<int32_t>(
    toptr,
    length);
}
ERROR awkward_new_Identities64(
  int64_t* toptr,
  int64_t length) {
  return awkward_new_Identities<int64_t>(
    toptr,
    length);
}

ERROR awkward_Identities32_to_Identities64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  int64_t width) {
  for (int64_t i = 0;  i < length*width;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}

template <typename ID, typename T>
ERROR awkward_Identities_from_ListOffsetArray(
  ID* toptr,
  const ID* fromptr,
  const T* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  int64_t globalstart = fromoffsets[0];
  int64_t globalstop = fromoffsets[fromlength];
  for (int64_t k = 0;  k < globalstart*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  for (int64_t k = globalstop*(fromwidth + 1);
       k < tolength*(fromwidth + 1);
       k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    int64_t start = fromoffsets[i];
    int64_t stop = fromoffsets[i + 1];
    if (start != stop  &&  stop > tolength) {
      return failure("max(stop) > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = start;  j < stop;  j++) {
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[j*(fromwidth + 1) + k] =
          fromptr[i*(fromwidth) + k];
      }
      toptr[j*(fromwidth + 1) + fromwidth] = (ID)(j - start);
    }
  }
  return success();
}
ERROR awkward_Identities32_from_ListOffsetArray32(
  int32_t* toptr,
  const int32_t* fromptr,
  const int32_t* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListOffsetArray<int32_t, int32_t>(
    toptr,
    fromptr,
    fromoffsets,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities32_from_ListOffsetArrayU32(
  int32_t* toptr,
  const int32_t* fromptr,
  const uint32_t* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListOffsetArray<int32_t, uint32_t>(
    toptr,
    fromptr,
    fromoffsets,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities32_from_ListOffsetArray64(
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListOffsetArray<int32_t, int64_t>(
    toptr,
    fromptr,
    fromoffsets,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_ListOffsetArray32(
  int64_t* toptr,
  const int64_t* fromptr,
  const int32_t* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListOffsetArray<int64_t, int32_t>(
    toptr,
    fromptr,
    fromoffsets,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_ListOffsetArrayU32(
  int64_t* toptr,
  const int64_t* fromptr,
  const uint32_t* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListOffsetArray<int64_t, uint32_t>(
    toptr,
    fromptr,
    fromoffsets,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_ListOffsetArray64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* fromoffsets,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListOffsetArray<int64_t, int64_t>(
    toptr,
    fromptr,
    fromoffsets,
    tolength,
    fromlength,
    fromwidth);
}

template <typename ID, typename T>
ERROR awkward_Identities_from_ListArray(
  bool* uniquecontents,
  ID* toptr,
  const ID* fromptr,
  const T* fromstarts,
  const T* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  for (int64_t k = 0;  k < tolength*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    int64_t start = fromstarts[i];
    int64_t stop = fromstops[i];
    if (start != stop  &&  stop > tolength) {
      return failure("max(stop) > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
    for (int64_t j = start;  j < stop;  j++) {
      if (toptr[j*(fromwidth + 1) + fromwidth] != -1) {
        *uniquecontents = false;
        return success();   // calling code won't use the (incomplete) toptr
      }                     // if there are any non-unique contents
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[j*(fromwidth + 1) + k] =
          fromptr[i*(fromwidth) + k];
      }
      toptr[j*(fromwidth + 1) + fromwidth] = (ID)(j - start);
    }
  }
  *uniquecontents = true;
  return success();
}
ERROR awkward_Identities32_from_ListArray32(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListArray<int32_t, int32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities32_from_ListArrayU32(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListArray<int32_t, uint32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities32_from_ListArray64(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListArray<int32_t, int64_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_ListArray32(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListArray<int64_t, int32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_ListArrayU32(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListArray<int64_t, uint32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_ListArray64(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_ListArray<int64_t, int64_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromstarts,
    fromstops,
    tolength,
    fromlength,
    fromwidth);
}

template <typename ID>
ERROR awkward_Identities_from_RegularArray(
  ID* toptr,
  const ID* fromptr,
  int64_t size,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  for (int64_t i = 0;  i < fromlength;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[(i*size + j)*(fromwidth + 1) + k] =
          fromptr[i*fromwidth + k];
      }
      toptr[(i*size + j)*(fromwidth + 1) + fromwidth] = (ID)(j);
    }
  }
  for (int64_t k = (fromlength + 1)*size*(fromwidth + 1);
       k < tolength*(fromwidth + 1);
       k++) {
    toptr[k] = -1;
  }
  return success();
}
ERROR awkward_Identities32_from_RegularArray(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t size,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_RegularArray<int32_t>(
    toptr,
    fromptr,
    size,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_RegularArray(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t size,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_RegularArray<int64_t>(
    toptr,
    fromptr,
    size,
    tolength,
    fromlength,
    fromwidth);
}

template <typename ID, typename T>
ERROR awkward_Identities_from_IndexedArray(
  bool* uniquecontents,
  ID* toptr,
  const ID* fromptr,
  const T* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  for (int64_t k = 0;  k < tolength*fromwidth;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    T j = fromindex[i];
    if (j >= tolength) {
      return failure("max(index) > len(content)", i, j, FILENAME(__LINE__));
    }
    else if (j >= 0) {
      if (toptr[j*fromwidth] != -1) {
        *uniquecontents = false;
        return success();   // calling code won't use the (incomplete) toptr
      }                     // if there are any non-unique contents
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[j*fromwidth + k] = fromptr[i*fromwidth + k];
      }
    }
  }
  *uniquecontents = true;
  return success();
}
ERROR awkward_Identities32_from_IndexedArray32(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_IndexedArray<int32_t, int32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities32_from_IndexedArrayU32(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const uint32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_IndexedArray<int32_t, uint32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities32_from_IndexedArray64(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_IndexedArray<int32_t, int64_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_IndexedArray32(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_IndexedArray<int64_t, int32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_IndexedArrayU32(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const uint32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_IndexedArray<int64_t, uint32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    tolength,
    fromlength,
    fromwidth);
}
ERROR awkward_Identities64_from_IndexedArray64(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth) {
  return awkward_Identities_from_IndexedArray<int64_t, int64_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromindex,
    tolength,
    fromlength,
    fromwidth);
}

template <typename ID, typename T, typename I>
ERROR awkward_Identities_from_UnionArray(
  bool* uniquecontents,
  ID* toptr,
  const ID* fromptr,
  const T* fromtags,
  const I* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  for (int64_t k = 0;  k < tolength*fromwidth;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    if (fromtags[i] == which) {
      I j = fromindex[i];
      if (j >= tolength) {
        return failure("max(index) > len(content)", i, j, FILENAME(__LINE__));
      }
      else if (j < 0) {
        return failure("min(index) < 0", i, j, FILENAME(__LINE__));
      }
      else {
        if (toptr[j*fromwidth] != -1) {
          *uniquecontents = false;
          return success();   // calling code won't use the (incomplete) toptr
        }                     // if there are any non-unique contents
        for (int64_t k = 0;  k < fromwidth;  k++) {
          toptr[j*fromwidth + k] = fromptr[i*fromwidth + k];
        }
      }
    }
  }
  *uniquecontents = true;
  return success();
}
ERROR awkward_Identities32_from_UnionArray8_32(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  return awkward_Identities_from_UnionArray<int32_t, int8_t, int32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    tolength,
    fromlength,
    fromwidth,
    which);
}
ERROR awkward_Identities32_from_UnionArray8_U32(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  return awkward_Identities_from_UnionArray<int32_t, int8_t, uint32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    tolength,
    fromlength,
    fromwidth,
    which);
}
ERROR awkward_Identities32_from_UnionArray8_64(
  bool* uniquecontents,
  int32_t* toptr,
  const int32_t* fromptr,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  return awkward_Identities_from_UnionArray<int32_t, int8_t, int64_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    tolength,
    fromlength,
    fromwidth,
    which);
}
ERROR awkward_Identities64_from_UnionArray8_32(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  return awkward_Identities_from_UnionArray<int64_t, int8_t, int32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    tolength,
    fromlength,
    fromwidth,
    which);
}
ERROR awkward_Identities64_from_UnionArray8_U32(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  return awkward_Identities_from_UnionArray<int64_t, int8_t, uint32_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    tolength,
    fromlength,
    fromwidth,
    which);
}
ERROR awkward_Identities64_from_UnionArray8_64(
  bool* uniquecontents,
  int64_t* toptr,
  const int64_t* fromptr,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t tolength,
  int64_t fromlength,
  int64_t fromwidth,
  int64_t which) {
  return awkward_Identities_from_UnionArray<int64_t, int8_t, int64_t>(
    uniquecontents,
    toptr,
    fromptr,
    fromtags,
    fromindex,
    tolength,
    fromlength,
    fromwidth,
    which);
}

template <typename ID>
ERROR awkward_Identities_extend(
  ID* toptr,
  const ID* fromptr,
  int64_t fromlength,
  int64_t tolength) {
  int64_t i = 0;
  for (;  i < fromlength;  i++) {
    toptr[i] = fromptr[i];
  }
  for (;  i < tolength;  i++) {
    toptr[i] = -1;
  }
  return success();
}
ERROR awkward_Identities32_extend(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t fromlength,
  int64_t tolength) {
  return awkward_Identities_extend<int32_t>(
    toptr,
    fromptr,
    fromlength,
    tolength);
}
ERROR awkward_Identities64_extend(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t fromlength,
  int64_t tolength) {
  return awkward_Identities_extend<int64_t>(
    toptr,
    fromptr,
    fromlength,
    tolength);
}
