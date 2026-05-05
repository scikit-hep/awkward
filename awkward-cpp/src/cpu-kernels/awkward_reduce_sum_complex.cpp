// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_sum_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT real = static_cast<OUT>(0);
    OUT imag = static_cast<OUT>(0);
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      real += static_cast<OUT>(fromptr[i * 2]);
      imag += static_cast<OUT>(fromptr[i * 2 + 1]);
    }
    toptr[bin * 2] = real;
    toptr[bin * 2 + 1] = imag;
  }
  return success();
}
ERROR awkward_reduce_sum_complex64_complex64_64(
  float* toptr,
  const float* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  return awkward_reduce_sum_complex<float, float>(toptr, fromptr, offsets, outlength);
}
ERROR awkward_reduce_sum_complex128_complex128_64(
  double* toptr,
  const double* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  return awkward_reduce_sum_complex<double, double>(toptr, fromptr, offsets, outlength);
}
