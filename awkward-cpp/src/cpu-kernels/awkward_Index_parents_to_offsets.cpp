// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Index_parents_to_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_Index_parents_to_offsets(
  T* tooffsets,
  const T* fromparents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0; i <= outlength; i++) {
    tooffsets[i] = 0;
  }

  for (int64_t i = 0; i < lenparents; i++) {
    int64_t p = fromparents[i];
    if (p < outlength) {
      tooffsets[p + 1]++;
    }
  }

  for (int64_t i = 1; i <= outlength; i++) {
    tooffsets[i] += tooffsets[i - 1];
  }

  return success();
}
ERROR awkward_Index_parents_to_offsets_64(
  int64_t* tooffsets,
  const int64_t* fromparents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_Index_parents_to_offsets<int64_t>(
    tooffsets,
    fromparents,
    lenparents,
    outlength);
}