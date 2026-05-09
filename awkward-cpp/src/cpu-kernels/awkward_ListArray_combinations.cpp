// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_combinations.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

template <typename C>
ERROR awkward_ListArray_combinations(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const C* starts,
  const C* stops,
  int64_t length) {
  for (int64_t j = 0;  j < n;  j++) {
    toindex[j] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = (int64_t)starts[i];
    int64_t stop = (int64_t)stops[i];
    fromindex[0] = start;
    awkward_ListArray_combinations_step_64(
      tocarry,
      toindex,
      fromindex,
      0,
      stop,
      n,
      replacement);
  }
  return success();
}

#define WRAPPER(SUFFIX, C) \
  ERROR awkward_ListArray##SUFFIX(int64_t** tocarry, int64_t* toindex, int64_t* fromindex, int64_t n, bool replacement, const C* starts, const C* stops, int64_t length) { \
    return awkward_ListArray_combinations<C>(tocarry, toindex, fromindex, n, replacement, starts, stops, length); \
  }

WRAPPER(32_combinations_64, int32_t)
WRAPPER(U32_combinations_64, uint32_t)
WRAPPER(64_combinations_64, int64_t)
