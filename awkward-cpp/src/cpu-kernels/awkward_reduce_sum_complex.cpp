// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_sum_complex(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
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

#define WRAPPER(FUNC, OUT, IN) \
  ERROR FUNC(OUT* toptr, const IN* fromptr, const int64_t* offsets, int64_t outlength) { \
    return awkward_reduce_sum_complex<OUT, IN>(toptr, fromptr, offsets, outlength); \
  }

WRAPPER(awkward_reduce_sum_complex64_complex64_64, float, float)
WRAPPER(awkward_reduce_sum_complex128_complex128_64, double, double)
