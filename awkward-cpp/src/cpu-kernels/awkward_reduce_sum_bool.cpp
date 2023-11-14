// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_bool.cpp", line)

#include "awkward/kernels.h"

template <typename IN>
ERROR awkward_reduce_sum_bool(
  bool* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = false;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] |= (fromptr[i] != 0);
  }
  return success();
}
ERROR awkward_reduce_sum_bool_bool_64(
  bool* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<bool>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_int8_64(
  bool* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_uint8_64(
  bool* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_int16_64(
  bool* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_uint16_64(
  bool* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_int32_64(
  bool* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_uint32_64(
  bool* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_int64_64(
  bool* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_uint64_64(
  bool* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_float32_64(
  bool* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_bool_float64_64(
  bool* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum_bool<double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
