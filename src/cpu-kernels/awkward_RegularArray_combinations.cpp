// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_combinations.cpp", line)

#include "awkward/kernels.h"

template <typename T>
void awkward_ListArray_combinations_step(
  T** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t j,
  int64_t stop,
  int64_t n,
  bool replacement) {
  while (fromindex[j] < stop) {
    if (replacement) {
      for (int64_t k = j + 1;  k < n;  k++) {
        fromindex[k] = fromindex[j];
      }
    }
    else {
      for (int64_t k = j + 1;  k < n;  k++) {
        fromindex[k] = fromindex[j] + (k - j);
      }
    }
    if (j + 1 == n) {
      for (int64_t k = 0;  k < n;  k++) {
        tocarry[k][toindex[k]] = fromindex[k];
        toindex[k]++;
      }
    }
    else {
      awkward_ListArray_combinations_step<T>(
        tocarry,
        toindex,
        fromindex,
        j + 1,
        stop,
        n,
        replacement);
    }
    fromindex[j]++;
  }
}

template <typename C, typename T>
ERROR awkward_RegularArray_combinations(
  T** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  int64_t size,
  int64_t length) {
  for (int64_t j = 0;  j < n;  j++) {
    toindex[j] = 0;
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = size*i;
    int64_t stop = start + size;
    fromindex[0] = start;
    awkward_ListArray_combinations_step<T>(
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
ERROR awkward_RegularArray_combinations_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t n,
  bool replacement,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_combinations<int32_t, int64_t>(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    size,
    length);
}
