// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_jagged_expand.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_jagged_expand(
  T* multistarts,
  T* multistops,
  const T* singleoffsets,
  int64_t regularsize,
  int64_t regularlength) {
  for (int64_t i = 0;  i < regularlength;  i++) {
    for (int64_t j = 0;  j < regularsize;  j++) {
      multistarts[i*regularsize + j] = singleoffsets[j];
      multistops[i*regularsize + j] = singleoffsets[j + 1];
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_jagged_expand_64(
  int64_t* multistarts,
  int64_t* multistops,
  const int64_t* singleoffsets,
  int64_t regularsize,
  int64_t regularlength) {
  return awkward_RegularArray_getitem_jagged_expand<int64_t>(
    multistarts,
    multistops,
    singleoffsets,
    regularsize,
    regularlength);
}
