// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_simplify.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN, typename TO>
ERROR awkward_IndexedArray_simplify(
  TO* toindex,
  const OUT* outerindex,
  int64_t outerlength,
  const IN* innerindex,
  int64_t innerlength) {
  for (int64_t i = 0;  i < outerlength;  i++) {
    OUT j = outerindex[i];
    if (j < 0) {
      toindex[i] = -1;
    }
    else if (j >= innerlength) {
      return failure("index out of range", i, j, FILENAME(__LINE__));
    }
    else {
      toindex[i] = innerindex[j];
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, OUT, IN, TO) \
  ERROR awkward_IndexedArray##SUFFIX(TO* toindex, const OUT* outerindex, int64_t outerlength, const IN* innerindex, int64_t innerlength) { \
    return awkward_IndexedArray_simplify<OUT, IN, TO>(toindex, outerindex, outerlength, innerindex, innerlength); \
  }

WRAPPER(32_simplify32_to64, int32_t, int32_t, int64_t)
WRAPPER(32_simplifyU32_to64, int32_t, uint32_t, int64_t)
WRAPPER(32_simplify64_to64, int32_t, int64_t, int64_t)
WRAPPER(U32_simplify32_to64, uint32_t, int32_t, int64_t)
WRAPPER(U32_simplifyU32_to64, uint32_t, uint32_t, int64_t)
WRAPPER(U32_simplify64_to64, uint32_t, int64_t, int64_t)
WRAPPER(64_simplify32_to64, int64_t, int32_t, int64_t)
WRAPPER(64_simplifyU32_to64, int64_t, uint32_t, int64_t)
WRAPPER(64_simplify64_to64, int64_t, int64_t, int64_t)
