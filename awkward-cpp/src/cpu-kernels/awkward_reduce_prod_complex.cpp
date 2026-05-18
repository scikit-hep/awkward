// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod_complex.cpp", line)

#include "awkward/kernels.h"

#include <complex>

template <typename OUT, typename IN>
ERROR awkward_reduce_prod_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT a = static_cast<OUT>(1);
    OUT b = static_cast<OUT>(0);
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      OUT c = static_cast<OUT>(fromptr[i * 2]);
      OUT d = static_cast<OUT>(fromptr[i * 2 + 1]);
      OUT na = a * c - b * d;
      OUT nb = a * d + b * c;
      a = na;
      b = nb;
    }
    toptr[bin * 2] = a;
    toptr[bin * 2 + 1] = b;
  }
  return success();
}

#define WRAPPER(SUFFIX, OUT, IN) \
  ERROR awkward_reduce_prod_complex##SUFFIX(OUT* toptr, const IN* fromptr, const int64_t* offsets, int64_t outlength) { \
    return awkward_reduce_prod_complex<OUT, IN>(toptr, fromptr, offsets, outlength); \
  }

WRAPPER(64_complex64_64, float, float)
WRAPPER(128_complex128_64, double, double)
