// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_filltags_const.cpp", line)

#include "awkward/kernels.h"

template <typename TO>
ERROR awkward_UnionArray_filltags_const(
  TO* totags,
  int64_t totagsoffset,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    totags[totagsoffset + i] = (TO)base;
  }
  return success();
}
ERROR awkward_UnionArray_filltags_to8_const(
  int8_t* totags,
  int64_t totagsoffset,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_filltags_const<int8_t>(
    totags,
    totagsoffset,
    length,
    base);
}
