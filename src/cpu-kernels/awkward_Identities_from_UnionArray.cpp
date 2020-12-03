// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities_from_UnionArray.cpp", line)

#include "awkward/kernels.h"

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
