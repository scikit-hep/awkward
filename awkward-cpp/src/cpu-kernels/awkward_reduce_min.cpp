// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_min.cpp", line)

#include "awkward/kernels.h"

template <typename OUT, typename IN>
ERROR awkward_reduce_min(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  OUT identity) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = identity;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    IN x = fromptr[i];
    toptr[parents[i]] = (x < toptr[parents[i]] ? x : toptr[parents[i]]);
  }
  return success();
}
ERROR awkward_reduce_min_int8_int8_64(
  int8_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int8_t identity) {
  return awkward_reduce_min<int8_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_uint8_uint8_64(
  uint8_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint8_t identity) {
  return awkward_reduce_min<uint8_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_int16_int16_64(
  int16_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int16_t identity) {
  return awkward_reduce_min<int16_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_uint16_uint16_64(
  uint16_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint16_t identity) {
  return awkward_reduce_min<uint16_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_int32_int32_64(
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int32_t identity) {
  return awkward_reduce_min<int32_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_uint32_uint32_64(
  uint32_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint32_t identity) {
  return awkward_reduce_min<uint32_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_int64_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int64_t identity) {
  return awkward_reduce_min<int64_t, int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_uint64_uint64_64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint64_t identity) {
  return awkward_reduce_min<uint64_t, uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_float32_float32_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  float identity) {
  return awkward_reduce_min<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_min_float64_float64_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  double identity) {
  return awkward_reduce_min<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
