// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities_from_IndexedArray.cpp", line)

#include "awkward/kernels.h"

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
