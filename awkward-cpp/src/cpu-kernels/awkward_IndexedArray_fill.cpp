// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_fill.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR awkward_IndexedArray_fill(
  TO* __restrict__ toindex,
  int64_t toindexoffset,
  const FROM* __restrict__ fromindex,
  int64_t length,
  int64_t base) {
  for (int64_t i = 0;  i < length;  i++) {
    FROM fromval = fromindex[i];
    toindex[toindexoffset + i] = fromval < 0 ? -1 : (TO)(fromval + base);
  }
  return success();
}

#define WRAPPER(FUNC, FROM, TO) \
  ERROR FUNC(TO* toindex, int64_t toindexoffset, const FROM* fromindex, int64_t length, int64_t base) { \
    return awkward_IndexedArray_fill<FROM, TO>(toindex, toindexoffset, fromindex, length, base); \
  }

WRAPPER(awkward_IndexedArray_fill_to64_from32, int32_t, int64_t)
WRAPPER(awkward_IndexedArray_fill_to64_fromU32, uint32_t, int64_t)
WRAPPER(awkward_IndexedArray_fill_to64_from64, int64_t, int64_t)
