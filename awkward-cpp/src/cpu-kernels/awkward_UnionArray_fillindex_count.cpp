// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_fillindex_count.cpp", line)

#include "awkward/kernels.h"

template <typename TO>
ERROR awkward_UnionArray_fillindex_count(
  TO* toindex,
  int64_t toindexoffset,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = (TO)i;
  }
  return success();
}
ERROR awkward_UnionArray_fillindex_to64_count(
  int64_t* toindex,
  int64_t toindexoffset,
  int64_t length) {
  return awkward_UnionArray_fillindex_count<int64_t>(
    toindex,
    toindexoffset,
    length);
}
