// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_array_regularize.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_array_regularize(
  T* toarray,
  const T* fromarray,
  int64_t lenarray,
  int64_t size) {
  for (int64_t j = 0;  j < lenarray;  j++) {
    toarray[j] = fromarray[j];
    if (toarray[j] < 0) {
      toarray[j] += size;
    }
    if (!(0 <= toarray[j]  &&  toarray[j] < size)) {
      return failure("index out of range", kSliceNone, fromarray[j], FILENAME(__LINE__));
    }
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_regularize_64(
  int64_t* toarray,
  const int64_t* fromarray,
  int64_t lenarray,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array_regularize<int64_t>(
    toarray,
    fromarray,
    lenarray,
    size);
}
