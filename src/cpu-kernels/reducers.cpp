// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>
#include <vector>

#include "awkward/cpu-kernels/reducers.h"

ERROR awkward_reduce_count_64(int64_t* toptr, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]]++;
  }
  return success();
}

template <typename IN>
ERROR awkward_reduce_countnonzero(int64_t* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] += (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_countnonzero_bool_64(int64_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<bool>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_int8_64(int64_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_uint8_64(int64_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_int16_64(int64_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_uint16_64(int64_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_int32_64(int64_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_uint32_64(int64_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_uint64_64(int64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_float32_64(int64_t* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_countnonzero_float64_64(int64_t* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_countnonzero<double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_sum(OUT* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (OUT)0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] += (OUT)fromptr[fromptroffset + i];
  }
  return success();
}
ERROR awkward_reduce_sum_int64_bool_64(int64_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] += (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_sum_int64_int8_64(int64_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int64_t, int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint64_uint8_64(uint64_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_int64_int16_64(int64_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int64_t, int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint64_uint16_64(uint64_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_int64_int32_64(int64_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int64_t, int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint64_uint32_64(uint64_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_int64_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int64_t, int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint64_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint64_t, uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_float32_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<float, float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_float64_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<double, double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_int32_bool_64(int32_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] += (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_sum_int32_int8_64(int32_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int32_t, int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint32_uint8_64(uint32_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint32_t, uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_int32_int16_64(int32_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int32_t, int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint32_uint16_64(uint32_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint32_t, uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_int32_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<int32_t, int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_uint32_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum<uint32_t, uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}

template <typename IN>
ERROR awkward_reduce_sum_bool(bool* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (bool)0;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] |= (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_sum_bool_bool_64(bool* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<bool>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_int8_64(bool* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_uint8_64(bool* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_int16_64(bool* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_uint16_64(bool* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_int32_64(bool* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_uint32_64(bool* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_int64_64(bool* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_uint64_64(bool* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_float32_64(bool* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_sum_bool_float64_64(bool* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_sum_bool<double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_prod(OUT* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (OUT)1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] *= (OUT)fromptr[fromptroffset + i];
  }
  return success();
}
ERROR awkward_reduce_prod_int64_bool_64(int64_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] *= (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_prod_int64_int8_64(int64_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int64_t, int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint64_uint8_64(uint64_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int64_int16_64(int64_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int64_t, int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint64_uint16_64(uint64_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int64_int32_64(int64_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int64_t, int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint64_uint32_64(uint64_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int64_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int64_t, int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint64_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint64_t, uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_float32_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<float, float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_float64_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<double, double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int32_bool_64(int32_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] *= (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_prod_int32_int8_64(int32_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int32_t, int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint32_uint8_64(uint32_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int32_int16_64(int32_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int32_t, int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint32_uint16_64(uint32_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_int32_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<int32_t, int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_uint32_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod<uint32_t, uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}

template <typename IN>
ERROR awkward_reduce_prod_bool(bool* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = (bool)1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] &= (fromptr[fromptroffset + i] != 0);
  }
  return success();
}
ERROR awkward_reduce_prod_bool_bool_64(bool* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<bool>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_int8_64(bool* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_uint8_64(bool* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_int16_64(bool* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_uint16_64(bool* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_int32_64(bool* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_uint32_64(bool* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_int64_64(bool* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_uint64_64(bool* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_float32_64(bool* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_prod_bool_float64_64(bool* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_prod_bool<double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_min(OUT* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, OUT identity) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = identity;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    IN x = fromptr[fromptroffset + i];
    toptr[parents[parentsoffset + i]] = (x < toptr[parents[parentsoffset + i]] ? x : toptr[parents[parentsoffset + i]]);
  }
  return success();
}
ERROR awkward_reduce_min_int8_int8_64(int8_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int8_t identity) {
  return awkward_reduce_min<int8_t, int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_uint8_uint8_64(uint8_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint8_t identity) {
  return awkward_reduce_min<uint8_t, uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_int16_int16_64(int16_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int16_t identity) {
  return awkward_reduce_min<int16_t, int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_uint16_uint16_64(uint16_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint16_t identity) {
  return awkward_reduce_min<uint16_t, uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_int32_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int32_t identity) {
  return awkward_reduce_min<int32_t, int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_uint32_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint32_t identity) {
  return awkward_reduce_min<uint32_t, uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_int64_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int64_t identity) {
  return awkward_reduce_min<int64_t, int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_uint64_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint64_t identity) {
  return awkward_reduce_min<uint64_t, uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_float32_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, float identity) {
  return awkward_reduce_min<float, float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_min_float64_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, double identity) {
  return awkward_reduce_min<double, double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_max(OUT* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, OUT identity) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = identity;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    IN x = fromptr[fromptroffset + i];
    toptr[parents[parentsoffset + i]] = (x > toptr[parents[parentsoffset + i]] ? x : toptr[parents[parentsoffset + i]]);
  }
  return success();
}
ERROR awkward_reduce_max_int8_int8_64(int8_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int8_t identity) {
  return awkward_reduce_max<int8_t, int8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_uint8_uint8_64(uint8_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint8_t identity) {
  return awkward_reduce_max<uint8_t, uint8_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_int16_int16_64(int16_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int16_t identity) {
  return awkward_reduce_max<int16_t, int16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_uint16_uint16_64(uint16_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint16_t identity) {
  return awkward_reduce_max<uint16_t, uint16_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_int32_int32_64(int32_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int32_t identity) {
  return awkward_reduce_max<int32_t, int32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_uint32_uint32_64(uint32_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint32_t identity) {
  return awkward_reduce_max<uint32_t, uint32_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_int64_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, int64_t identity) {
  return awkward_reduce_max<int64_t, int64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_uint64_uint64_64(uint64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, uint64_t identity) {
  return awkward_reduce_max<uint64_t, uint64_t>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_float32_float32_64(float* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, float identity) {
  return awkward_reduce_max<float, float>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}
ERROR awkward_reduce_max_float64_float64_64(double* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength, double identity) {
  return awkward_reduce_max<double, double>(toptr, fromptr, fromptroffset, parents, parentsoffset, lenparents, outlength, identity);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_argmin(OUT* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[parentsoffset + i];
    int64_t start = starts[parent];
    if (toptr[parent] == -1  ||  fromptr[fromptroffset + i] < fromptr[fromptroffset + toptr[parent] + start]) {
      toptr[parent] = i - start;
    }
  }
  return success();
}
ERROR awkward_reduce_argmin_bool_64(int64_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[parentsoffset + i];
    int64_t start = starts[parent];
    if (toptr[parent] == -1  ||  (fromptr[fromptroffset + i] != 0) < (fromptr[fromptroffset + toptr[parent] + start] != 0)) {
      toptr[parent] = i - start;
    }
  }
  return success();
}
ERROR awkward_reduce_argmin_int8_64(int64_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int8_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_uint8_64(int64_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint8_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_int16_64(int64_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int16_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_uint16_64(int64_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint16_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_int32_64(int64_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int32_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_uint32_64(int64_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint32_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, int64_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_uint64_64(int64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, uint64_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_float32_64(int64_t* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, float>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmin_float64_64(int64_t* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmin<int64_t, double>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}

template <typename OUT, typename IN>
ERROR awkward_reduce_argmax(OUT* toptr, const IN* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[parentsoffset + i];
    int64_t start = starts[parent];
    if (toptr[parent] == -1  ||  fromptr[fromptroffset + i] > fromptr[fromptroffset + toptr[parent] + start]) {
      toptr[parent] = i - start;
    }
  }
  return success();
}
ERROR awkward_reduce_argmax_bool_64(int64_t* toptr, const bool* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = -1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[parentsoffset + i];
    int64_t start = starts[parent];
    if (toptr[parent] == -1  ||  (fromptr[fromptroffset + i] != 0) > (fromptr[fromptroffset + toptr[parent] + start] != 0)) {
      toptr[parent] = i - start;
    }
  }
  return success();
}
ERROR awkward_reduce_argmax_int8_64(int64_t* toptr, const int8_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int8_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_uint8_64(int64_t* toptr, const uint8_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint8_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_int16_64(int64_t* toptr, const int16_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int16_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_uint16_64(int64_t* toptr, const uint16_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint16_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_int32_64(int64_t* toptr, const int32_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int32_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_uint32_64(int64_t* toptr, const uint32_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint32_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_int64_64(int64_t* toptr, const int64_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, int64_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_uint64_64(int64_t* toptr, const uint64_t* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, uint64_t>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_float32_64(int64_t* toptr, const float* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, float>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}
ERROR awkward_reduce_argmax_float64_64(int64_t* toptr, const double* fromptr, int64_t fromptroffset, const int64_t* starts, int64_t startsoffset, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  return awkward_reduce_argmax<int64_t, double>(toptr, fromptr, fromptroffset, starts, startsoffset, parents, parentsoffset, lenparents, outlength);
}

ERROR awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toparents[i] = 0;
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_global_startstop_64(int64_t* globalstart, int64_t* globalstop, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  *globalstart = offsets[offsetsoffset + 0];
  *globalstop = offsets[offsetsoffset + length];
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_maxcount_offsetscopy_64(int64_t* maxcount, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  *maxcount = 0;
  offsetscopy[0] = offsets[offsetsoffset + 0];
  for (int64_t i = 0;  i < length;  i++) {
    int64_t count = offsets[offsetsoffset + i + 1] - offsets[offsetsoffset + i];
    if (*maxcount < count) {
      *maxcount = count;
    }
    offsetscopy[i + 1] = offsets[offsetsoffset + i + 1];
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_preparenext_64(int64_t* nextcarry, int64_t* nextparents, int64_t nextlen, int64_t* maxnextparents, int64_t* distincts, int64_t distinctslen, int64_t* offsetscopy, const int64_t* offsets, int64_t offsetsoffset, int64_t length, const int64_t* parents, int64_t parentsoffset, int64_t maxcount) {
  *maxnextparents = 0;
  for (int64_t i = 0;  i < distinctslen;  i++) {
    distincts[i] = -1;
  }

  int64_t k = 0;
  while (k < nextlen) {
    int64_t j = 0;
    for (int64_t i = 0;  i < length;  i++) {
      if (offsetscopy[i] < offsets[offsetsoffset + i + 1]) {
        int64_t count = offsets[offsetsoffset + i + 1] - offsets[offsetsoffset + i];
        int64_t diff = offsetscopy[i] - offsets[offsetsoffset + i];
        int64_t parent = parents[parentsoffset + i];

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

ERROR awkward_listoffsetarray_reduce_nonlocal_nextstarts_64(int64_t* nextstarts, const int64_t* nextparents, int64_t nextlen) {
  int64_t lastnextparent = -1;
  for (int64_t k = 0;  k < nextlen;  k++) {
    if (nextparents[k] != lastnextparent) {
      nextstarts[nextparents[k]] = k;
    }
    lastnextparent = nextparents[k];
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_findgaps_64(int64_t* gaps, const int64_t* parents, int64_t parentsoffset, int64_t lenparents) {
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    int64_t parent = parents[parentsoffset + i];
    if (last < parent) {
      gaps[k] = parent - last;
      k++;
      last = parent;
    }
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_nonlocal_outstartsstops_64(int64_t* outstarts, int64_t* outstops, const int64_t* distincts, int64_t lendistincts, const int64_t* gaps) {
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
  return success();
}

ERROR awkward_listoffsetarray_reduce_local_nextparents_64(int64_t* nextparents, const int64_t* offsets, int64_t offsetsoffset, int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    for (int64_t j = offsets[offsetsoffset + i];  j < offsets[offsetsoffset + i + 1];  j++) {
      nextparents[j] = i;
    }
  }
  return success();
}

ERROR awkward_listoffsetarray_reduce_local_outoffsets_64(int64_t* outoffsets, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  outoffsets[outlength] = lenparents;
  int64_t k = 0;
  int64_t last = -1;
  for (int64_t i = 0;  i < lenparents;  i++) {
    while (last < parents[parentsoffset + i]) {
      outoffsets[k] = i;
      k++;
      last++;
    }
  }
  return success();
}

template <typename T>
ERROR awkward_indexedarray_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, int64_t* outindex, const T* index, int64_t indexoffset, const int64_t* parents, int64_t parentsoffset, int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (index[indexoffset + i] >= 0) {
      nextcarry[k] = index[indexoffset + i];
      nextparents[k] = parents[parentsoffset + i];
      outindex[i] = k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
ERROR awkward_indexedarray32_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, int64_t* outindex, const int32_t* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length) {
  return awkward_indexedarray_reduce_next_64<int32_t>(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length);
}
ERROR awkward_indexedarrayU32_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, int64_t* outindex, const uint32_t* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length) {
  return awkward_indexedarray_reduce_next_64<uint32_t>(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length);
}
ERROR awkward_indexedarray64_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, int64_t* outindex, const int64_t* index, int64_t indexoffset, int64_t* parents, int64_t parentsoffset, int64_t length) {
  return awkward_indexedarray_reduce_next_64<int64_t>(nextcarry, nextparents, outindex, index, indexoffset, parents, parentsoffset, length);
}

ERROR awkward_indexedarray_reduce_next_fix_offsets_64(int64_t* outoffsets, const int64_t* starts, int64_t startsoffset, int64_t startslength, int64_t outindexlength) {
  for (int64_t i = 0;  i < startslength;  i++) {
    outoffsets[i] = starts[startsoffset + i];
  }
  outoffsets[startsoffset + startslength] = outindexlength;
  return success();
}

ERROR awkward_numpyarray_reduce_mask_bytemaskedarray(int8_t* toptr, const int64_t* parents, int64_t parentsoffset, int64_t lenparents, int64_t outlength) {
  for (int64_t i = 0;  i < outlength;  i++) {
    toptr[i] = 1;
  }
  for (int64_t i = 0;  i < lenparents;  i++) {
    toptr[parents[parentsoffset + i]] = 0;
  }
  return success();
}

ERROR awkward_bytemaskedarray_reduce_next_64(int64_t* nextcarry, int64_t* nextparents, int64_t* outindex, const int8_t* mask, int64_t maskoffset, const int64_t* parents, int64_t parentsoffset, int64_t length, bool validwhen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if ((mask[maskoffset + i] != 0) == validwhen) {
      nextcarry[k] = i;
      nextparents[k] = parents[parentsoffset + i];
      outindex[i] = k;
      k++;
    }
    else {
      outindex[i] = -1;
    }
  }
  return success();
}
