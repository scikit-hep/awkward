// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmax_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_argmax_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t k = 0;  k < outlength;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (toptr[parent] == -1  ||
      (fromptr[i * 2] > fromptr[toptr[parent] * 2]  ||
        (fromptr[i * 2] == fromptr[toptr[parent] * 2]  &&
         fromptr[i * 2 + 1] > fromptr[toptr[parent] * 2 + 1]))) {
      toptr[parent] = i;
    }
  }
  return success();
}
ERROR awkward_reduce_argmax_complex64_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax_complex<int64_t, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_complex128_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax_complex<int64_t, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
