// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_prod(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (OUT)1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] *= (OUT)fromptr[i];
  }
  return success();
}
ERROR awkward_reduce_prod_int64_int8_64(
  int64_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint64_uint8_64(
  uint64_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_int64_int16_64(
  int64_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint64_uint16_64(
  uint64_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_int64_int32_64(
  int64_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint64_uint32_64(
  uint64_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_int64_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int64_t, int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint64_uint64_64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_float32_float32_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_float64_float64_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_int32_int8_64(
  int32_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int32_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint32_uint8_64(
  uint32_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_int32_int16_64(
  int32_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int32_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint32_uint16_64(
  uint32_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_int32_int32_64(
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<int32_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_uint32_uint32_64(
  uint32_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
