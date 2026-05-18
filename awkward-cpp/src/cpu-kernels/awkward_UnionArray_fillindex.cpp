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

#define WRAPPER(SUFFIX, FROM, TO) \
  ERROR awkward_UnionArray_fillindex_to64_from##SUFFIX(TO* toindex, int64_t toindexoffset, const FROM* fromindex, int64_t length) { \
    return awkward_UnionArray_fillindex<FROM, TO>(toindex, toindexoffset, fromindex, length); \
  }

WRAPPER(32, int32_t, int64_t)
WRAPPER(U32, uint32_t, int64_t)
WRAPPER(64, int64_t, int64_t)
