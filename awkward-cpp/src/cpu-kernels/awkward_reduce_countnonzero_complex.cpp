// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_countnonzero_complex.cpp", line)

#include "awkward/kernels.h"

template <typename IN>
ERROR awkward_reduce_countnonzero_complex(
  int64_t* toptr,
  const IN* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    int64_t c = 0;
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      if (fromptr[i * 2] != 0 || fromptr[i * 2 + 1] != 0) c++;
    }
    toptr[bin] = c;
  }
  return success();
}
ERROR awkward_reduce_countnonzero_complex64_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  return awkward_reduce_countnonzero_complex<float>(toptr, fromptr, offsets, outlength);
}
ERROR awkward_reduce_countnonzero_complex128_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* offsets,
  int64_t outlength) {
  return awkward_reduce_countnonzero_complex<double>(toptr, fromptr, offsets, outlength);
}
