// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities_from_ListOffsetArray.cpp", line)

#include "awkward/kernels.h"

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
