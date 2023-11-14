// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Index_nones_as_index.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_Index_nones_as_index(
  T* toindex,
  int64_t length) {
  int64_t n_non_null = 0;
  // Assuming that `toindex` comprises of unique, contiguous integers (or -1), and is zero-based
  // Compute the number of non-null values to determine our starting index
  for (int64_t i = 0; i < length; i++) {
    if (toindex[i] != -1) {
        n_non_null++;
    }
  }
  // Now set the null-value indices to by monotonically increasing and unique from the final index
  for (int64_t i = 0; i < length; i++) {
    toindex[i] == -1 ? toindex[i] = n_non_null++ : toindex[i];
  }
  return success();
}
ERROR awkward_Index_nones_as_index_64(
  int64_t* toindex,
  int64_t length) {
  return awkward_Index_nones_as_index<int64_t>(
    toindex,
    length);
}
