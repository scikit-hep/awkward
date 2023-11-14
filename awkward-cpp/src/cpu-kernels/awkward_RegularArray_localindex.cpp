// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_localindex.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_localindex(
  T* toindex,
  int64_t size,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = 0;  j < size;  j++) {
      toindex[i*size + j] = j;
    }
  }
  return success();
}
ERROR awkward_RegularArray_localindex_64(
  int64_t* toindex,
  int64_t size,
  int64_t length) {
  return awkward_RegularArray_localindex<int64_t>(
    toindex,
    size,
    length);
}
