// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_fill_count.cpp", line)

#include "awkward/kernels.h"

template <typename TO>
ERROR awkward_IndexedArray_fill_count(
  TO* toindex,
  int64_t toindexoffset,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[toindexoffset + i] = i + base;
  }
  return success();
}
ERROR awkward_IndexedArray_fill_to64_count(
  int64_t* toindex,
  int64_t toindexoffset,
  int64_t length,
  int64_t base) {
  return awkward_IndexedArray_fill_count(
    toindex,
    toindexoffset,
    length,
    base);
}
