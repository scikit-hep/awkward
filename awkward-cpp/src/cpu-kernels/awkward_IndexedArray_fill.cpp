// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_fill.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR awkward_IndexedArray_fill(
  TO* toindex,
  int64_t toindexoffset,
  const FROM* fromindex,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    FROM fromval = fromindex[i];
    toindex[toindexoffset + i] = fromval < 0 ? -1 : (TO)(fromval + base);
  }
  return success();
}
ERROR awkward_IndexedArray_fill_to64_from32(
  int64_t* toindex,
  int64_t toindexoffset,
  const int32_t* fromindex,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill<int32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length,
    base);
}
ERROR awkward_IndexedArray_fill_to64_fromU32(
  int64_t* toindex,
  int64_t toindexoffset,
  const uint32_t* fromindex,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill<uint32_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length,
    base);
}
ERROR awkward_IndexedArray_fill_to64_from64(
  int64_t* toindex,
  int64_t toindexoffset,
  const int64_t* fromindex,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill<int64_t, int64_t>(
    toindex,
    toindexoffset,
    fromindex,
    length,
    base);
}
