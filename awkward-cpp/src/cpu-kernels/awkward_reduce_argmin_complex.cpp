// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmin_complex.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_argmin_complex(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {

  std::fill_n(toptr, outlength, static_cast<OUT>(-1));

  for (int64_t i = 0; i < lenparents; i++) {
    int64_t parent = parents[i];
    int64_t current_idx = toptr[parent];

    if (current_idx == -1) {
      toptr[parent] = i;
    } else {
      IN real_i = fromptr[i * 2];
      IN imag_i = fromptr[i * 2 + 1];
      IN real_min = fromptr[current_idx * 2];
      IN imag_min = fromptr[current_idx * 2 + 1];

      if (real_i < real_min || (real_i == real_min && imag_i < imag_min)) {
        toptr[parent] = i;
      }
    }
  }
  return success();
}
ERROR awkward_reduce_argmin_complex64_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin_complex<int64_t, float>(
    toptr,
    fromptr,
    parents,
    offsets,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_complex128_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin_complex<int64_t, double>(
    toptr,
    fromptr,
    parents,
    offsets,
    lenparents,
    outlength);
}
