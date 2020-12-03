// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities_from_RegularArray.cpp", line)

#include "awkward/kernels.h"

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
