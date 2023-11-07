// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_fill.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR awkward_ListArray_fill(
  TO* tostarts,
  int64_t tostartsoffset,
  TO* tostops,
  int64_t tostopsoffset,
  const FROM* fromstarts,
  const FROM* fromstops,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    tostarts[tostartsoffset + i] = (TO)(fromstarts[i] + base);
    tostops[tostopsoffset + i] = (TO)(fromstops[i] + base);
  }
  return success();
}
ERROR awkward_ListArray_fill_to64_from32(
  int64_t* tostarts,
  int64_t tostartsoffset,
  int64_t* tostops,
  int64_t tostopsoffset,
  const int32_t* fromstarts,
  const int32_t* fromstops,
  int64_t length,
  int64_t base) {
  return awkward_ListArray_fill<int32_t, int64_t>(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base);
}
ERROR awkward_ListArray_fill_to64_fromU32(
  int64_t* tostarts,
  int64_t tostartsoffset,
  int64_t* tostops,
  int64_t tostopsoffset,
  const uint32_t* fromstarts,
  const uint32_t* fromstops,
  int64_t length,
  int64_t base) {
  return awkward_ListArray_fill<uint32_t, int64_t>(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base);
}
ERROR awkward_ListArray_fill_to64_from64(
  int64_t* tostarts,
  int64_t tostartsoffset,
  int64_t* tostops,
  int64_t tostopsoffset,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t base) {
  return awkward_ListArray_fill<int64_t, int64_t>(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base);
}
