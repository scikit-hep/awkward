// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_combinations.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_combinations(
  T* toindex,
  int64_t n,
  bool replacement,
  int64_t singlelen) {
  return failure("FIXME: awkward_combinations", 0, kSliceNone, FILENAME(__LINE__));
}
ERROR awkward_combinations_64(
  int64_t* toindex,
  int64_t n,
  bool replacement,
  int64_t singlelen) {
  return awkward_combinations<int64_t>(
    toindex,
    n,
    replacement,
    singlelen);
}
