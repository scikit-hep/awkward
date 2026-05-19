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

#define WRAPPER(SUFFIX, FROM, TO) \
  ERROR awkward_IndexedArray_fill_to64_from##SUFFIX(TO* toindex, int64_t toindexoffset, const FROM* fromindex, int64_t length, int64_t base) { \
    return awkward_IndexedArray_fill<FROM, TO>(toindex, toindexoffset, fromindex, length, base); \
  }

WRAPPER(32, int32_t, int64_t)
WRAPPER(U32, uint32_t, int64_t)
WRAPPER(64, int64_t, int64_t)
