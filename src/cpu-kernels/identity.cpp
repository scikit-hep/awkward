// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"

template <typename T>
ERROR awkward_new_identity(T* toptr, int64_t length) {
  for (T i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_new_identity32(int32_t* toptr, int64_t length) {
  return awkward_new_identity<int32_t>(toptr, length);
}
ERROR awkward_new_identity64(int64_t* toptr, int64_t length) {
  return awkward_new_identity<int64_t>(toptr, length);
}

ERROR awkward_identity32_to_identity64(int64_t* toptr, const int32_t* fromptr, int64_t length, int64_t width) {
  for (int64_t i = 0;  i < length*width;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}

template <typename ID, typename T>
ERROR awkward_identity_from_listoffsetarray(ID* toptr, const ID* fromptr, const T* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  int64_t globalstart = fromoffsets[offsetsoffset];
  int64_t globalstop = fromoffsets[offsetoffset + fromlength];
  for (int64_t k = 0;  k < globalstart*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  for (int64_t k = globalstop*(fromwidth + 1);  k < tolength*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    int64_t start = fromoffsets[offsetsoffset + i];
    int64_t stop = fromoffsets[offsetsoffset + i + 1];
    if (start != stop  &&  stop > tolength) {
      return failure("max(stop) > len(content)", i, kSliceNone);
    }
    for (int64_t j = start;  j < stop;  j++) {
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[j*(fromwidth + 1) + k] = fromptr[fromptroffset + i*(fromwidth) + k];
      }
      toptr[j*(fromwidth + 1) + fromwidth] = ID(j - start);
    }
  }
  return success();
}
ERROR awkward_identity32_from_listoffsetarray32(int32_t* toptr, const int32_t* fromptr, const int32_t* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listoffsetarray<int32_t, int32_t>(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_listoffsetarray32(int64_t* toptr, const int64_t* fromptr, const int32_t* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listoffsetarray<int64_t, int32_t>(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_listoffsetarrayU32(int64_t* toptr, const int64_t* fromptr, const uint32_t* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listoffsetarray<int64_t, uint32_t>(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_listoffsetarray64(int64_t* toptr, const int64_t* fromptr, const int64_t* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listoffsetarray<int64_t, int64_t>(toptr, fromptr, fromoffsets, fromptroffset, offsetsoffset, tolength, fromlength, fromwidth);
}

template <typename ID, typename T>
ERROR awkward_identity_from_listarray(ID* toptr, const ID* fromptr, const T* fromstarts, const T* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  for (int64_t k = 0;  k < tolength*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    int64_t start = fromstarts[startsoffset + i];
    int64_t stop = fromstops[stopsoffset + i];
    if (start != stop  &&  stop > tolength) {
      return failure("max(stop) > len(content)", i, kSliceNone);
    }
    for (int64_t j = start;  j < stop;  j++) {
      if (toptr[j*(fromwidth + 1) + fromwidth] != -1) {
        return failure("item has ambiguous identity", i, kSliceNone);
      }
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[j*(fromwidth + 1) + k] = fromptr[fromptroffset + i*(fromwidth) + k];
      }
      toptr[j*(fromwidth + 1) + fromwidth] = ID(j - start);
    }
  }
  return success();
}
ERROR awkward_identity32_from_listarray32(int32_t* toptr, const int32_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int32_t, int32_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_listarray32(int64_t* toptr, const int64_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int64_t, int32_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_listarrayU32(int64_t* toptr, const int64_t* fromptr, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int64_t, uint32_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_listarray64(int64_t* toptr, const int64_t* fromptr, const int64_t* fromstarts, const int64_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int64_t, int64_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}

template <typename ID>
ERROR awkward_identity_from_regulararray(ID* toptr, const ID* fromptr, int64_t fromptroffset, int64_t size, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  for (int64_t i = 0;  i < fromlength;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      for (int64_t k = 0;  k < fromwidth;  k++) {
        toptr[(i*size + j)*(fromwidth + 1) + k] = fromptr[fromptroffset + i*fromwidth + k];
      }
      toptr[(i*size + j)*(fromwidth + 1) + fromwidth] = ID(j);
    }
  }
  for (int64_t k = (fromlength + 1)*size*(fromwidth + 1);  k < tolength*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  return success();
}
ERROR awkward_identity32_from_regulararray(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, int64_t size, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_regulararray<int32_t>(toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth);
}
ERROR awkward_identity64_from_regulararray(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, int64_t size, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_regulararray<int64_t>(toptr, fromptr, fromptroffset, size, tolength, fromlength, fromwidth);
}
