// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#include "awkward/kernels.h"
#include <complex>

template <typename T>
ERROR
awkward_reduce_centered_sumofsquares_complex(T* toptr,
                                             const T* fromptr,
                                             const int64_t* offsets,
                                             int64_t outlength,
                                             const T* means) {
  for (int64_t bin = 0; bin < outlength; ++bin) {
    T acc{};
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; ++i) {
      T a = fromptr[2 * i] - means[2 * bin];
      T b = fromptr[2 * i + 1] - means[2 * bin + 1];
      acc += a * a + b * b;
    }
    toptr[bin] = acc;
  }
  return success();
}

ERROR
awkward_reduce_centered_sumofsquares_complex_64_64(float* toptr,
                                                   const float* fromptr,
                                                   const int64_t* offsets,
                                                   int64_t outlength,
                                                   const float* means) {
  return awkward_reduce_centered_sumofsquares_complex(
      toptr, fromptr, offsets, outlength, means);
}

ERROR
awkward_reduce_centered_sumofsquares_complex_128_64(double* toptr,
                                                    const double* fromptr,
                                                    const int64_t* offsets,
                                                    int64_t outlength,
                                                    const double* means) {
  return awkward_reduce_centered_sumofsquares_complex(
      toptr, fromptr, offsets, outlength, means);
}

ERROR
awkward_reduce_centered_sumofsquares_complex_256_64(long double* toptr,
                                                    const long double* fromptr,
                                                    const int64_t* offsets,
                                                    int64_t outlength,
                                                    const long double* means) {
  return awkward_reduce_centered_sumofsquares_complex(
      toptr, fromptr, offsets, outlength, means);
}
