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
ERROR awkward_ListArray32_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const int32_t* starts,
  const int32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations<int32_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArrayU32_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const uint32_t* starts,
  const uint32_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations<uint32_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length);
}
ERROR awkward_ListArray64_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  const int64_t* starts,
  const int64_t* stops,
  int64_t length) {
  return awkward_ListArray_combinations<int64_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length);
}
