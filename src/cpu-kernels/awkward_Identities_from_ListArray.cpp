// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities_from_ListArray.cpp", line)

#include "awkward/kernels.h"

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
