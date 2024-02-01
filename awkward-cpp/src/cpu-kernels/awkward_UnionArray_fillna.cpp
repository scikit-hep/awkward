// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_fillna.cpp", line)

#include "awkward/kernels.h"

template <typename T, typename C>
ERROR awkward_UnionArray_fillna(
  T* toindex,
  const C* fromindex,
  int64_t length) {
  for (int64_t i = 0; i < length; i++)
  {
    toindex[i] = fromindex[i] >= 0 ? fromindex[i] : 0;
  }
  return success();
}
ERROR awkward_UnionArray_fillna_from32_to64(
  int64_t* toindex,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillna<int64_t, int32_t>(
    toindex,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillna_fromU32_to64(
  int64_t* toindex,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillna<int64_t, uint32_t>(
    toindex,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillna_from64_to64(
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillna<int64_t, int64_t>(
    toindex,
    fromindex,
    length);
}
