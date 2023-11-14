// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_fillindex.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR awkward_UnionArray_fillindex(
  TO* toindex,
  int64_t toindexoffset,
  const FROM* fromindex,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = (TO)fromindex[i];
  }
  return success();
}
ERROR awkward_UnionArray_fillindex_to64_from32(
  int64_t* toindex,
  int64_t toindexoffset,
  const int32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillindex<int32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillindex_to64_fromU32(
  int64_t* toindex,
  int64_t toindexoffset,
  const uint32_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillindex<uint32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length);
}
ERROR awkward_UnionArray_fillindex_to64_from64(
  int64_t* toindex,
  int64_t toindexoffset,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_UnionArray_fillindex<int64_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length);
}
