// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#include "awkward/kernels.h"
#include <complex>

template <typename T>
ERROR
awkward_reduce_sumofsquares_complex(T* toptr,
                                    const T* fromptr,
                                    const int64_t* offsets,
                                    int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; ++bin) {
    T real{}, imag{};
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; ++i) {
      T a = fromptr[2 * i], b = fromptr[2 * i + 1];
      real += a * a - b * b;
      imag += a * b + a * b;
    }
    toptr[2 * bin] = real;
    toptr[2 * bin + 1] = imag;
  }
  return success();
}

ERROR
awkward_reduce_sumofsquares_complex_64_64(float* toptr,
                                          const float* fromptr,
                                          const int64_t* offsets,
                                          int64_t outlength) {
  return awkward_reduce_sumofsquares_complex(
      toptr, fromptr, offsets, outlength);
}

ERROR
awkward_reduce_sumofsquares_complex_128_64(double* toptr,
                                           const double* fromptr,
                                           const int64_t* offsets,
                                           int64_t outlength) {
  return awkward_reduce_sumofsquares_complex(
      toptr, fromptr, offsets, outlength);
}

ERROR
awkward_reduce_sumofsquares_complex_256_64(long double* toptr,
                                           const long double* fromptr,
                                           const int64_t* offsets,
                                           int64_t outlength) {
  return awkward_reduce_sumofsquares_complex(
      toptr, fromptr, offsets, outlength);
}
