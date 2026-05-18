// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_missing_repeat.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_missing_repeat(
  T* outindex,
  const T* index,
  int64_t indexlength,
  int64_t repetitions,
  int64_t regularsize) {
  for (int64_t i = 0;  i < repetitions;  i++) {
    int64_t out_offset = i * indexlength;
    int64_t val_offset = i * regularsize;

    for (int64_t j = 0;  j < indexlength;  j++) {
      T base = index[j];
      T adjustment = (base >= 0) ? val_offset : 0;
      outindex[out_offset + j] = base + adjustment;
    }
  }
  return success();
}
ERROR awkward_missing_repeat_64(
  int64_t* outindex,
  const int64_t* index,
  int64_t indexlength,
  int64_t repetitions,
  int64_t regularsize) {
  return awkward_missing_repeat<int64_t>(
    outindex,
    index,
    indexlength,
    repetitions,
    regularsize);
}
