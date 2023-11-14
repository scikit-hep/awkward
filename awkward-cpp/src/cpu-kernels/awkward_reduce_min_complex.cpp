// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_min_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_min_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  OUT identity) {
  for (int64_t i = 0;  i < outlength;  i++) {
   toptr[i * 2] = identity;
   toptr[i * 2 + 1] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    IN x = fromptr[i * 2];
    IN y = fromptr[i * 2 + 1];
    if (x < toptr[parent * 2]  ||
      (x == toptr[parent * 2]  &&  y < toptr[parent * 2 + 1])) {
      toptr[parent * 2] = x;
      toptr[parent * 2 + 1] = y;
    }
  }
  return success();
}
ERROR awkward_reduce_min_complex64_complex64_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  float identity) {
  return awkward_reduce_min_complex<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_complex128_complex128_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  double identity) {
  return awkward_reduce_min_complex<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
