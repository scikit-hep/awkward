// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities_extend.cpp", line)

#include "awkward/kernels.h"

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
