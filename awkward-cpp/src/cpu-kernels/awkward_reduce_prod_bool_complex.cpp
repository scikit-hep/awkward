// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod_bool_complex.cpp", line)

#include "awkward/kernels.h"

template <typename IN>
ERROR awkward_reduce_prod_bool_complex(
  bool* toptr,
  const IN* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  std::memset(toptr, 1, outlength * sizeof(bool));

  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] &= (fromptr[i * 2] != 0  ||  fromptr[i * 2 + 1] != 0);
  }
  for (int64_t i = 0; i < lenparents; ++i) {
    bool condition = (fromptr[i * 2] != 0) | (fromptr[i * 2 + 1] != 0);
    toptr[parents[i]] &= condition;
}
  return success();
}
ERROR awkward_reduce_prod_bool_complex64_64(
  bool* toptr,
  const float* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool_complex<float>(
    toptr,
    fromptr,
    parents,
    offsets,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_complex128_64(
  bool* toptr,
  const double* fromptr,
  const int64_t* parents,
  const int64_t* offsets,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool_complex<double>(
    toptr,
    fromptr,
    parents,
    offsets,
    lenparents,
    outlength);
}
