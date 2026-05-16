// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_max_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_max_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  int64_t outlength,
  OUT identity) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT best_re = identity;
    OUT best_im = 0;
    bool seen = false;
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      IN x = fromptr[i * 2];
      IN y = fromptr[i * 2 + 1];
      if (!seen || x > best_re || (x == best_re && y > best_im)) {
        best_re = static_cast<OUT>(x);
        best_im = static_cast<OUT>(y);
        seen = true;
      }
    }
    toptr[bin * 2] = best_re;
    toptr[bin * 2 + 1] = best_im;
  }
  return success();
}

#define WRAPPER(SUFFIX, OUT, IN) \
  ERROR awkward_reduce_max_complex##SUFFIX(OUT* toptr, const IN* fromptr, const int64_t* offsets, int64_t outlength, OUT identity) { \
    return awkward_reduce_max_complex<OUT, IN>(toptr, fromptr, offsets, outlength, identity); \
  }

WRAPPER(64_complex64_64, float, float)
WRAPPER(128_complex128_64, double, double)
