// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/kernel-utils.cpp", line)

#include "awkward/kernel-utils.h"

void awkward_regularize_rangeslice(
  int64_t* start,
  int64_t* stop,
  bool posstep,
  bool hasstart,
  bool hasstop,
  int64_t length) {
  if (posstep) {
    if (!hasstart)           *start = 0;
    else if (*start < 0)     *start += length;
    if (*start < 0)          *start = 0;
    if (*start > length)     *start = length;

    if (!hasstop)            *stop = length;
    else if (*stop < 0)      *stop += length;
    if (*stop < 0)           *stop = 0;
    if (*stop > length)      *stop = length;
    if (*stop < *start)      *stop = *start;
  }

  else {
    if (!hasstart)           *start = length - 1;
    else if (*start < 0)     *start += length;
    if (*start < -1)         *start = -1;
    if (*start > length - 1) *start = length - 1;

    if (!hasstop)            *stop = -1;
    else if (*stop < 0)      *stop += length;
    if (*stop < -1)          *stop = -1;
    if (*stop > length - 1)  *stop = length - 1;
    if (*stop > *start)      *stop = *start;
  }
}

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
        tocarry[k][toindex[k]] = (T)fromindex[k];
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

void awkward_ListArray_combinations_step_64(
  int64_t** tocarry,
  int64_t* toindex,
  int64_t* fromindex,
  int64_t j,
  int64_t stop,
  int64_t n,
  bool replacement) {
    return awkward_ListArray_combinations_step<int64_t>(
      tocarry,
      toindex,
      fromindex,
      j,
      stop,
      n,
      replacement
    );
}
