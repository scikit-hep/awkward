// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"

template <typename T>
Error awkward_new_identity(T* toptr, int64_t length) {
  for (T i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
Error awkward_new_identity32(int32_t* toptr, int64_t length) {
  return awkward_new_identity<int32_t>(toptr, length);
}
Error awkward_new_identity64(int64_t* toptr, int64_t length) {
  return awkward_new_identity<int64_t>(toptr, length);
}

Error awkward_identity32_to_identity64(int64_t* toptr, const int32_t* fromptr, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}

template <typename ID, typename T>
Error awkward_identity_from_listarray(ID* toptr, const ID* fromptr, const T* fromstarts, const T* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  for (int64_t k = 0;  k < tolength*(fromwidth + 1);  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < fromlength;  i++) {
    int64_t start = fromstarts[startsoffset + i];
    int64_t stop = fromstops[stopsoffset + i];
    if (start != stop  &&  stop > tolength) {
      return failure(i, -1, "max(stop) > len(content)");
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
Error awkward_identity32_from_listarray32(int32_t* toptr, const int32_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int32_t, int32_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}
Error awkward_identity64_from_listarray32(int64_t* toptr, const int64_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int64_t, int32_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}
Error awkward_identity64_from_listarray64(int64_t* toptr, const int64_t* fromptr, const int64_t* fromstarts, const int64_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth) {
  return awkward_identity_from_listarray<int64_t, int64_t>(toptr, fromptr, fromstarts, fromstops, fromptroffset, startsoffset, stopsoffset, tolength, fromlength, fromwidth);
}
