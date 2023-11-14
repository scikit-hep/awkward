// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_filltags.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR awkward_UnionArray_filltags(
  TO* totags,
  int64_t totagsoffset,
  const FROM* fromtags,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    totags[totagsoffset + i] = (TO)(fromtags[i] + base);
  }
  return success();
}
ERROR awkward_UnionArray_filltags_to8_from8(
  int8_t* totags,
  int64_t totagsoffset,
  const int8_t* fromtags,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_filltags<int8_t, int8_t>(
    totags,
    totagsoffset,
    fromtags,
    length,
    base);
}
