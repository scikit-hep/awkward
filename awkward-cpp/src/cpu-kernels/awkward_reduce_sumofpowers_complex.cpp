// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#include "awkward/kernels.h"
#include <cmath>
#include <limits>

template <typename T>
ERROR
awkward_reduce_sumofpowers_complex(T* toptr,
                                   const T* fromptr,
                                   const int64_t* offsets,
                                   int64_t outlength,
                                   int64_t n) {
  for (int64_t bin = 0; bin < outlength; ++bin) {
    T real{}, imag{};
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; ++i) {
      T br = fromptr[2 * i], bi = fromptr[2 * i + 1];
      if (n == 1) {
        real += br;
        imag += bi;
        continue;
      }
      if (n < 0) {
        if (br == 0 && bi == 0) {
          br = std::numeric_limits<T>::quiet_NaN();
          bi = br;
        } else if (std::abs(br) >= std::abs(bi)) {
          T ratio = bi / br;
          T denom = br + bi * ratio;
          br = T{1} / denom;
          bi = -ratio / denom;
        } else {
          T ratio = br / bi;
          T denom = bi + br * ratio;
          br = ratio / denom;
          bi = -T{1} / denom;
        }
      }
      uint64_t power = n < 0 ? uint64_t(-(n + 1)) + 1 : uint64_t(n);
      T pr{1}, pi{};
      while (power != 0) {
        if (power & 1) {
          T next_real = pr * br - pi * bi;
          pi = pr * bi + pi * br;
          pr = next_real;
        }
        power >>= 1;
        if (power != 0) {
          T next_real = br * br - bi * bi;
          bi = T{2} * br * bi;
          br = next_real;
        }
      }
      real += pr;
      imag += pi;
    }
    toptr[2 * bin] = real;
    toptr[2 * bin + 1] = imag;
  }
  return success();
}

ERROR
awkward_reduce_sumofpowers_complex_64_64(float* toptr,
                                         const float* fromptr,
                                         const int64_t* offsets,
                                         int64_t outlength,
                                         int64_t n) {
  return awkward_reduce_sumofpowers_complex(
      toptr, fromptr, offsets, outlength, n);
}

ERROR
awkward_reduce_sumofpowers_complex_128_64(double* toptr,
                                          const double* fromptr,
                                          const int64_t* offsets,
                                          int64_t outlength,
                                          int64_t n) {
  return awkward_reduce_sumofpowers_complex(
      toptr, fromptr, offsets, outlength, n);
}

ERROR
awkward_reduce_sumofpowers_complex_256_64(long double* toptr,
                                          const long double* fromptr,
                                          const int64_t* offsets,
                                          int64_t outlength,
                                          int64_t n) {
  return awkward_reduce_sumofpowers_complex(
      toptr, fromptr, offsets, outlength, n);
}
