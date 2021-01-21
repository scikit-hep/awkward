// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod_complex.cpp", line)

#include "awkward/kernels.h"

// Let 'z1 = p + iq' and 'z2 = r + is' be two complex numbers (p, q, r and s are real),
// then their product 'z1*z2' is defined as 'z1*z2 = (pr - qs) + i(ps + qr)'.
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
    toptr[parents[i] * 2] = toptr[parents[i] * 2] * (OUT)fromptr[i * 2]
                          - toptr[parents[i] * 2 + 1] * (OUT)fromptr[i * 2 + 1];
    toptr[parents[i] * 2 + 1] = toptr[parents[i] * 2 ] * (OUT)fromptr[i * 2 + 1]
                              + toptr[parents[i] * 2 + 1] * (OUT)fromptr[i * 2];
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
