// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_localindex.cpp", line)

#include "awkward/kernels.h"
#include <numeric>

template <typename T>
ERROR awkward_localindex(
  T* toindex,
  int64_t length) {
  std::iota(toindex, toindex + length, 0);
  return success();
}
ERROR awkward_localindex_64(
  int64_t* toindex,
  int64_t length) {
  return awkward_localindex<int64_t>(
    toindex,
    length);
}
