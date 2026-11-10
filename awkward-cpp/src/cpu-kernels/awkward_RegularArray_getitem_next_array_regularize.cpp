// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_RegularArray_getitem_next_array_regularize.cpp", line)

#include "awkward/kernels.h"
#include "awkward/kernel-utils.h"

template <typename T>
ERROR awkward_RegularArray_getitem_next_array_regularize(
  T* __restrict__ toarray,
  const T* __restrict__ fromarray,
  int64_t lenarray,
  int64_t size) {
  for (int64_t j = 0;  j < lenarray;  j++) {
    const auto regular_at = awkward::kernel::normalize_index(fromarray[j], size);
    if (!regular_at) {
      // The reference writes the normalized failing index before returning.
      toarray[j] = fromarray[j] < 0 ? fromarray[j] + size : fromarray[j];
      return failure("index out of range", kSliceNone, fromarray[j], FILENAME(__LINE__));
    }
    toarray[j] = *regular_at;
  }
  return success();
}
ERROR awkward_RegularArray_getitem_next_array_regularize_64(
  int64_t* __restrict__ toarray,
  const int64_t* __restrict__ fromarray,
  int64_t lenarray,
  int64_t size) {
  return awkward_RegularArray_getitem_next_array_regularize<int64_t>(
    toarray,
    fromarray,
    lenarray,
    size);
}
