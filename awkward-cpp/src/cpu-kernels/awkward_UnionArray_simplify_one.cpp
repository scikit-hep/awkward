// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_UnionArray_simplify_one.cpp", line)

#include "awkward/kernels.h"

template <typename FROMTAGS,
          typename FROMINDEX,
          typename TOTAGS,
          typename TOINDEX>
ERROR awkward_UnionArray_simplify_one(
  TOTAGS* totags,
  TOINDEX* toindex,
  const FROMTAGS* fromtags,
  const FROMINDEX* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    if (fromtags[i] == fromwhich) {
      totags[i] = (TOTAGS)towhich;
      toindex[i] = (TOINDEX)(fromindex[i] + base);
    }
  }
  return success();
}
ERROR awkward_UnionArray8_32_simplify_one_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* fromtags,
  const int32_t* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify_one<int8_t, int32_t, int8_t, int64_t>(
    totags,
    toindex,
    fromtags,
    fromindex,
    towhich,
    fromwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_U32_simplify_one_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* fromtags,
  const uint32_t* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify_one<int8_t, uint32_t, int8_t, int64_t>(
    totags,
    toindex,
    fromtags,
    fromindex,
    towhich,
    fromwhich,
    length,
    base);
}
ERROR awkward_UnionArray8_64_simplify_one_to8_64(
  int8_t* totags,
  int64_t* toindex,
  const int8_t* fromtags,
  const int64_t* fromindex,
  int64_t towhich,
  int64_t fromwhich,
  int64_t length,
  int64_t base) {
  return awkward_UnionArray_simplify_one<int8_t, int64_t, int8_t, int64_t>(
    totags,
    toindex,
    fromtags,
    fromindex,
    towhich,
    fromwhich,
    length,
    base);
}
