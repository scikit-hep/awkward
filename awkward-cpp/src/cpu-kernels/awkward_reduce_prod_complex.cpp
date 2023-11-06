// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod_complex.cpp", line)

#include "awkward/kernels.h"

#include <complex>

template <typename OUT, typename IN>
ERROR awkward_reduce_prod_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i * 2] = (OUT)1;
    toptr[i * 2 + 1] = (OUT)0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    std::complex<OUT> z =
      std::complex<OUT>(toptr[parent * 2], toptr[parent * 2 + 1])
      * std::complex<OUT>((OUT)fromptr[i * 2], (OUT)fromptr[i * 2 + 1]);

    toptr[parent * 2] = z.real();
    toptr[parent * 2 + 1] = z.imag();
  }

  return success();
}
ERROR awkward_reduce_prod_complex64_complex64_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_complex<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_complex128_complex128_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_complex<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
