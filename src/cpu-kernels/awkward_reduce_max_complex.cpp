// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_max_complex.cpp", line)

#include <complex>

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_max_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  OUT identity) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i * 2] = identity;
    toptr[i * 2 + 1] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    std::complex<IN> z(fromptr[i * 2], fromptr[i * 2 + 1]);
    IN rho = std::abs(z);
    std::complex<IN> z_i(toptr[parents[i] * 2], toptr[parents[i] * 2 + 1]);
    IN rho_i = std::abs(z_i);
    if (rho > rho_i) {
      toptr[parents[i] * 2] = z.real();
      toptr[parents[i] * 2 + 1] = z.imag();
    }
    else {
      toptr[parents[i] * 2] = z_i.real();
      toptr[parents[i] * 2 + 1] = z_i.imag();
    }
  }
  return success();
}
ERROR awkward_reduce_max_complex64_complex64_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  float identity) {
  return awkward_reduce_max_complex<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_complex128_complex128_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  double identity) {
  return awkward_reduce_max_complex<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
