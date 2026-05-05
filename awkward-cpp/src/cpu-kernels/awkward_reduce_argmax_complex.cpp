// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmax_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_argmax_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t best = -1;
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      if (best == -1) {
        best = i;
      } else {
        IN x = fromptr[i * 2];
        IN y = fromptr[i * 2 + 1];
        IN bx = fromptr[best * 2];
        IN by = fromptr[best * 2 + 1];
        if (x > bx || (x == bx && y > by)) {
          best = i;
        }
      }
    }
    toptr[bin] = static_cast<OUT>(best);
  }
  return success();
}
ERROR awkward_reduce_argmax_complex64_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  return awkward_reduce_argmax_complex<int64_t, float>(toptr, fromptr, offsets, outlength);
}
ERROR awkward_reduce_argmax_complex128_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  return awkward_reduce_argmax_complex<int64_t, double>(toptr, fromptr, offsets, outlength);
}
