// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_countnonzero_complex.cpp", line)

#include "awkward/kernels.h"

template <typename IN>
ERROR awkward_reduce_countnonzero_complex(
  int64_t* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  std::memset(toptr, 0, outlength * sizeof(int64_t));

  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] += (fromptr[i * 2] != 0  ||  fromptr[i * 2 + 1] != 0);
  }
  return success();
}
ERROR awkward_reduce_countnonzero_complex64_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero_complex<float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_complex128_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero_complex<double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
