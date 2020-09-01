// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/reducers.cpp", line)

#include "awkward/kernels/reducers.h"

ERROR awkward_reduce_count_64(
  int64_t* toptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]]++;
  }
  return success();
}

template <typename IN>
ERROR awkward_reduce_countnonzero(
  int64_t* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] += (fromptr[i] != 0);
  }
  return success();
}
ERROR awkward_reduce_countnonzero_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<bool>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_int8_64(
  int64_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_uint8_64(
  int64_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_int16_64(
  int64_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_uint16_64(
  int64_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_int32_64(
  int64_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_uint32_64(
  int64_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_uint64_64(
  int64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_float32_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_countnonzero_float64_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_countnonzero<double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_sum(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (OUT)0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] += (OUT)fromptr[i];
  }
  return success();
}
ERROR awkward_reduce_sum_int64_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] += (fromptr[i] != 0);
  }
  return success();
}
ERROR awkward_reduce_sum_int64_int8_64(
  int64_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int64_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint64_uint8_64(
  uint64_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_int64_int16_64(
  int64_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int64_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint64_uint16_64(
  uint64_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_int64_int32_64(
  int64_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int64_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint64_uint32_64(
  uint64_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_int64_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int64_t, int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint64_uint64_64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_float32_float32_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_float64_float64_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_int32_bool_64(
  int32_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] += (fromptr[i] != 0);
  }
  return success();
}
ERROR awkward_reduce_sum_int32_int8_64(
  int32_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int32_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint32_uint8_64(
  uint32_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint32_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_int32_int16_64(
  int32_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int32_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint32_uint16_64(
  uint32_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint32_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_int32_int32_64(
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<int32_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_sum_uint32_uint32_64(
  uint32_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_sum<uint32_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}

template <typename IN>
ERROR awkward_reduce_sum_bool(
  bool* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (bool)0;
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
ERROR awkward_reduce_prod_int64_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] *= (fromptr[i] != 0);
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
ERROR awkward_reduce_prod_int32_bool_64(
  int32_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] *= (fromptr[i] != 0);
  }
  return success();
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

template <typename IN>
ERROR awkward_reduce_prod_bool(
  bool* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (bool)1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] &= (fromptr[i] != 0);
  }
  return success();
}
ERROR awkward_reduce_prod_bool_bool_64(
  bool* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<bool>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_int8_64(
  bool* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_uint8_64(
  bool* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_int16_64(
  bool* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_uint16_64(
  bool* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_int32_64(
  bool* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_uint32_64(
  bool* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_int64_64(
  bool* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_uint64_64(
  bool* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_float32_64(
  bool* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_prod_bool_float64_64(
  bool* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_prod_bool<double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}

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

template <typename OUT, typename IN>
ERROR awkward_reduce_max(
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
    toptr[parents[i]] = (x > toptr[parents[i]] ? x : toptr[parents[i]]);
  }
  return success();
}
ERROR awkward_reduce_max_int8_int8_64(
  int8_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int8_t identity) {
  return awkward_reduce_max<int8_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_uint8_uint8_64(
  uint8_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint8_t identity) {
  return awkward_reduce_max<uint8_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_int16_int16_64(
  int16_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int16_t identity) {
  return awkward_reduce_max<int16_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_uint16_uint16_64(
  uint16_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint16_t identity) {
  return awkward_reduce_max<uint16_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_int32_int32_64(
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int32_t identity) {
  return awkward_reduce_max<int32_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_uint32_uint32_64(
  uint32_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint32_t identity) {
  return awkward_reduce_max<uint32_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_int64_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  int64_t identity) {
  return awkward_reduce_max<int64_t, int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_uint64_uint64_64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  uint64_t identity) {
  return awkward_reduce_max<uint64_t, uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_float32_float32_64(
  float* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  float identity) {
  return awkward_reduce_max<float, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}
ERROR awkward_reduce_max_float64_float64_64(
  double* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength,
  double identity) {
  return awkward_reduce_max<double, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength,
    identity);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_argmin(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t k = 0;  k < outlength;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (toptr[parent] == -1  ||  fromptr[i] < fromptr[toptr[parent]]) {
      toptr[parent] = i;
    }
  }
  return success();
}
ERROR awkward_reduce_argmin_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t k = 0;  k < outlength;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (toptr[parent] == -1  ||  (fromptr[i] != 0) < (fromptr[toptr[parent]] != 0)) {
      toptr[parent] = i;
    }
  }
  return success();
}
ERROR awkward_reduce_argmin_int8_64(
  int64_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_uint8_64(
  int64_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_int16_64(
  int64_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_uint16_64(
  int64_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_int32_64(
  int64_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_uint32_64(
  int64_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_uint64_64(
  int64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_float32_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmin_float64_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmin<int64_t, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_argmax(
  OUT* toptr,
  const IN* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t k = 0;  k < outlength;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (toptr[parent] == -1  ||  (fromptr[i] > (fromptr[toptr[parent]]))) {
      toptr[parent] = i;
    }
  }
  return success();
}
ERROR awkward_reduce_argmax_bool_64(
  int64_t* toptr,
  const bool* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t k = 0;  k < outlength;  k++) {
    toptr[k] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (toptr[parent] == -1  ||  (fromptr[i] != 0) > (fromptr[toptr[parent]] != 0)) {
      toptr[parent] = i;
    }
  }
  return success();
}
ERROR awkward_reduce_argmax_int8_64(
  int64_t* toptr,
  const int8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_uint8_64(
  int64_t* toptr,
  const uint8_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint8_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_int16_64(
  int64_t* toptr,
  const int16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_uint16_64(
  int64_t* toptr,
  const uint16_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint16_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_int32_64(
  int64_t* toptr,
  const int32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_uint32_64(
  int64_t* toptr,
  const uint32_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint32_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_int64_64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_uint64_64(
  int64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint64_t>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_float32_64(
  int64_t* toptr,
  const float* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, float>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}
ERROR awkward_reduce_argmax_float64_64(
  int64_t* toptr,
  const double* fromptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  return awkward_reduce_argmax<int64_t, double>(
    toptr,
    fromptr,
    parents,
    lenparents,
    outlength);
}

ERROR awkward_content_reduce_zeroparents_64(
  int64_t* toparents,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toparents[i] = 0;
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_global_startstop_64(
  int64_t* globalstart,
  int64_t* globalstop,
  const int64_t* offsets,
  int64_t length) {
  *globalstart = offsets[0];
  *globalstop = offsets[length];
  return success();
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
  int64_t* maxcount,
  int64_t* offsetscopy,
  const int64_t* offsets,
  int64_t length) {
  *maxcount = 0;
  offsetscopy[0] = offsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    int64_t count = (offsets[i + 1] - offsets[i]);
    if (*maxcount < count) {
      *maxcount = count;
    }
    offsetscopy[i + 1] = offsets[i + 1];
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t nextlen,
  int64_t* maxnextparents,
  int64_t* distincts,
  int64_t distinctslen,
  int64_t* offsetscopy,
  const int64_t* offsets,
  int64_t length,
  const int64_t* parents,
  int64_t maxcount) {
  *maxnextparents = 0;
  for (int64_t i = 0;  i < distinctslen;  i++) {
    distincts[i] = -1;
  }

  int64_t k = 0;
  while (k < nextlen) {
    int64_t j = 0;
    for (int64_t i = 0;  i < length;  i++) {
      if (offsetscopy[i] < offsets[i + 1]) {
        int64_t diff = offsetscopy[i] - offsets[i];
        int64_t parent = parents[i];

        nextcarry[k] = offsetscopy[i];
        nextparents[k] = parent*maxcount + diff;

        if (*maxnextparents < nextparents[k]) {
          *maxnextparents = nextparents[k];
        }

        if (distincts[nextparents[k]] == -1) {
          distincts[nextparents[k]] = j;
          j++;
        }

        k++;
        offsetscopy[i]++;
      }
    }
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
  int64_t* nextstarts,
  const int64_t* nextparents,
  int64_t nextlen) {
  int64_t lastnextparent = -1;
  for (int64_t i = 0;  i < nextlen;  i++) {
    if (nextparents[i] != lastnextparent) {
      nextstarts[nextparents[i]] = i;
    }
    lastnextparent = nextparents[i];
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
  int64_t* gaps,
  const int64_t* parents,
  int64_t lenparents) {
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[i];
    if (last < parent) {
      gaps[k] = parent - last;
      k++;
      last = parent;
    }
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
  int64_t* outstarts,
  int64_t* outstops,
  const int64_t* distincts,
  int64_t lendistincts,
  const int64_t* gaps,
  int64_t outlength) {
  int64_t j = 0;
  int64_t k = 0;
  int64_t maxdistinct = -1;
  for (int64_t i = 0;  i < lendistincts;  i++) {
    if (maxdistinct < distincts[i]) {
      maxdistinct = distincts[i];
      for (int64_t gappy = 0;  gappy < gaps[j];  gappy++) {
        outstarts[k] = i;
        outstops[k] = i;
        k++;
      }
      j++;
    }
    if (distincts[i] != -1) {
      outstops[k - 1] = i + 1;
    }
  }
  for (;  k < outlength;  k++) {
    outstarts[k] = lendistincts + 1;
    outstops[k] = lendistincts + 1;
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64(
  int64_t* nummissing,
  int64_t* missing,
  int64_t* nextshifts,
  const int64_t* offsets,
  int64_t length,
  const int64_t* starts,
  const int64_t* parents,
  int64_t maxcount,
  int64_t nextlen,
  const int64_t* nextcarry) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t start = offsets[i];
    int64_t stop = offsets[i + 1];
    int64_t count = stop - start;

    if (starts[parents[i]] == i) {
      for (int64_t k = 0;  k < maxcount;  k++) {
        nummissing[k] = 0;
      }
    }

    for (int64_t k = count;  k < maxcount;  k++) {
      nummissing[k]++;
    }

    for (int64_t j = 0;  j < count;  j++) {
      missing[start + j] = nummissing[j];
    }
  }

  for (int64_t j = 0;  j < nextlen;  j++) {
    nextshifts[j] = missing[nextcarry[j]];
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_local_nextparents_64(
  int64_t* nextparents,
  const int64_t* offsets,
  int64_t length) {
  int64_t initialoffset = offsets[0];
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = offsets[i] - initialoffset;
         j < offsets[i + 1] - initialoffset;
         j++) {
      nextparents[j] = i;
    }
  }
  return success();
}

ERROR awkward_ListOffsetArray_reduce_local_outoffsets_64(
  int64_t* outoffsets,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    while (last < parents[i]) {
      outoffsets[k] = i;
      k++;
      last++;
    }
  }
  while (k <= outlength) {
    outoffsets[k] = lenparents;
    k++;
  }
  return success();
}

template <typename T>
ERROR awkward_IndexedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const T* index,
  const int64_t* parents,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (index[i] >= 0) {
      nextcarry[k] = index[i];
      nextparents[k] = parents[i];
      outindex[i] = k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int32_t* index,
  int64_t* parents,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_64<int32_t>(
    nextcarry,
    nextparents,
    outindex,
    index,
    parents,
    length);
}
ERROR awkward_IndexedArrayU32_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const uint32_t* index,
  int64_t* parents,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_64<uint32_t>(
    nextcarry,
    nextparents,
    outindex,
    index,
    parents,
    length);
}
ERROR awkward_IndexedArray64_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int64_t* index,
  int64_t* parents,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_64<int64_t>(
    nextcarry,
    nextparents,
    outindex,
    index,
    parents,
    length);
}

template <typename T>
ERROR awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64(
  int64_t* nextshifts,
  const T* index,
  int64_t length) {
  int64_t nullsum = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (index[i] >= 0) {
      nextshifts[k] = nullsum;
      k++;
    }
    else {
      nullsum++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_64(
  int64_t* nextshifts,
  const int32_t* index,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64<int32_t>(
    nextshifts,
    index,
    length);
}
ERROR awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_64(
  int64_t* nextshifts,
  const uint32_t* index,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64<uint32_t>(
    nextshifts,
    index,
    length);
}
ERROR awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_64(
  int64_t* nextshifts,
  const int64_t* index,
  int64_t length) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64<int64_t>(
    nextshifts,
    index,
    length);
}

template <typename T>
ERROR awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const T* index,
  int64_t length,
  const int64_t* shifts) {
  int64_t nullsum = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (index[i] >= 0) {
      nextshifts[k] = shifts[i] + nullsum;
      k++;
    }
    else {
      nullsum++;
    }
  }
  return success();
}
ERROR awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const int32_t* index,
  int64_t length,
  const int64_t* shifts) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64<int32_t>(
    nextshifts,
    index,
    length,
    shifts);
}
ERROR awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const uint32_t* index,
  int64_t length,
  const int64_t* shifts) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64<uint32_t>(
    nextshifts,
    index,
    length,
    shifts);
}
ERROR awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const int64_t* index,
  int64_t length,
  const int64_t* shifts) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64<int64_t>(
    nextshifts,
    index,
    length,
    shifts);
}

ERROR awkward_IndexedArray_reduce_next_fix_offsets_64(
  int64_t* outoffsets,
  const int64_t* starts,
  int64_t startslength,
  int64_t outindexlength) {
  for (int64_t i = 0;  i < startslength;  i++) {
    outoffsets[i] = starts[i];
  }
  outoffsets[startslength] = outindexlength;
  return success();
}

ERROR awkward_NumpyArray_reduce_adjust_starts_64(
  int64_t* toptr,
  int64_t outlength,
  const int64_t* parents,
  const int64_t* starts) {
  for (int64_t k = 0;  k < outlength;  k++) {
    int64_t i = toptr[k];
    if (i >= 0) {
      int64_t parent = parents[i];
      int64_t start = starts[parent];
      toptr[k] += -start;
    }
  }
  return success();
}

ERROR awkward_NumpyArray_reduce_adjust_starts_shifts_64(
  int64_t* toptr,
  int64_t outlength,
  const int64_t* parents,
  const int64_t* starts,
  const int64_t* shifts) {
  for (int64_t k = 0;  k < outlength;  k++) {
    int64_t i = toptr[k];
    if (i >= 0) {
      int64_t parent = parents[i];
      int64_t start = starts[parent];
      toptr[k] += shifts[i] - start;
    }
  }
  return success();
}

ERROR awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
  int8_t* toptr,
  const int64_t* parents,
  int64_t lenparents,
  int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[i]] = 0;
  }
  return success();
}

ERROR awkward_ByteMaskedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextparents,
  int64_t* outindex,
  const int8_t* mask,
  const int64_t* parents,
  int64_t length,
  bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == validwhen) {
      nextcarry[k] = i;
      nextparents[k] = parents[i];
      outindex[i] = k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}

ERROR awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(
  int64_t* nextshifts,
  const int8_t* mask,
  int64_t length,
  bool valid_when) {
  int64_t nullsum = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == (valid_when != 0)) {
      nextshifts[k] = nullsum;
      k++;
    }
    else {
      nullsum++;
    }
  }
  return success();
}

ERROR awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const int8_t* mask,
  int64_t length,
  bool valid_when,
  const int64_t* shifts) {
  int64_t nullsum = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[i] != 0) == (valid_when != 0)) {
      nextshifts[k] = shifts[i] + nullsum;
      k++;
    }
    else {
      nullsum++;
    }
  }
  return success();
}
