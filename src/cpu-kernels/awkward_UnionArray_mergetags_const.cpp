// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_mergetags_const.cpp", line)

#include "awkward/kernels.h"

template <typename TO>
ERROR awkward_UnionArray_mergetags_const(
  TO* totags,
  int64_t* toindex,
  int64_t offset,
  const int64_t* fromoffsets,
  int64_t index,
  TO tag,
  int64_t* nextoffset) {
  int64_t start = fromoffsets[index];
  int64_t stop = fromoffsets[index + 1];
  int64_t diff = stop - start;
  int64_t next = offset;
  for (int64_t i = 0; i < diff; i++) {
    totags[next] = tag;
    toindex[next] = start + i;
    next++;
  }
  *nextoffset = next;
  return success();
}
ERROR awkward_UnionArray_mergetags_to8_const(
  int8_t* totags,
  int64_t* toindex,
  int64_t offset,
  const int64_t* fromoffsets,
  int64_t index,
  int8_t tag,
  int64_t* nextoffset) {
  return awkward_UnionArray_mergetags_const<int8_t>(
    totags,
    toindex,
    offset,
    fromoffsets,
    index,
    tag,
    nextoffset);
}
