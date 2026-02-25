// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod_complex.cpp", line)

#include "awkward/kernels.h"

#include <complex>

template <typename OUT, typename IN>
ERROR awkward_reduce_prod_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0; i < outlength; i++) {
    toptr[i * 2] = static_cast<OUT>(1);
    toptr[i * 2 + 1] = static_cast<OUT>(0);
  }

  for (int64_t i = 0; i < lenparents; i++) {
    int64_t parent = parents[i];
    int64_t parent_idx = parent * 2;
    int64_t from_idx = i * 2;

    OUT a = toptr[parent_idx];
    OUT b = toptr[parent_idx + 1];
    OUT c = static_cast<OUT>(fromptr[from_idx]);
    OUT d = static_cast<OUT>(fromptr[from_idx + 1]);

    toptr[parent_idx] = a * c - b * d;
    toptr[parent_idx + 1] = a * d + b * c;
  }

  return success();
}
ERROR awkward_reduce_prod_complex64_complex64_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_complex<float, float>(
    toptr,
    fromptr,
    parents,
    offsets,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_complex128_complex128_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_complex<double, double>(
    toptr,
    fromptr,
    parents,
    offsets,
    lenparents,
    outlength);
}
