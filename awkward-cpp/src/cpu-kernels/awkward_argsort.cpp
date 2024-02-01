// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_argsort.cpp", line)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

template <typename T>
bool argsort_order_ascending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l < r);
}

template <typename T>
bool argsort_order_descending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l > r);
}

template <typename T>
ERROR awkward_argsort(
  int64_t* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> result(length);
  std::iota(result.begin(), result.end(), 0);

  if (ascending  &&  stable) {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(result.begin(), offsets[i]);
      auto stop = std::next(result.begin(), offsets[i + 1]);
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return argsort_order_ascending<T>(fromptr[i1], fromptr[i2]);
      });
      std::transform(start, stop, start, [&](int64_t j) -> int64_t {
        return j - offsets[i];
      });
    }
  }
  else if (!ascending  &&  stable) {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(result.begin(), offsets[i]);
      auto stop = std::next(result.begin(), offsets[i + 1]);
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return argsort_order_descending<T>(fromptr[i1], fromptr[i2]);
      });
      std::transform(start, stop, start, [&](int64_t j) -> int64_t {
        return j - offsets[i];
      });
    }
  }
  else if (ascending  &&  !stable) {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(result.begin(), offsets[i]);
      auto stop = std::next(result.begin(), offsets[i + 1]);
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return argsort_order_ascending<T>(fromptr[i1], fromptr[i2]);
      });
      std::transform(start, stop, start, [&](int64_t j) -> int64_t {
        return j - offsets[i];
      });
    }
  }
  else {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(result.begin(), offsets[i]);
      auto stop = std::next(result.begin(), offsets[i + 1]);
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return argsort_order_descending<T>(fromptr[i1], fromptr[i2]);
      });
      std::transform(start, stop, start, [&](int64_t j) -> int64_t {
        return j - offsets[i];
      });
    }
  }
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = result[i];
  }

  return success();
}

ERROR awkward_argsort_bool(
  int64_t* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int8(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint8(
  int64_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int16(
  int64_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint16(
  int64_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int32(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint32(
  int64_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_uint64(
  int64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_float32(
  int64_t* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

ERROR awkward_argsort_float64(
  int64_t* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  return awkward_argsort<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable);
}

template <>
bool argsort_order_ascending(bool l, bool r)
{
  return l < r;
}

template <>
bool argsort_order_descending(bool l, bool r)
{
  return l > r;
}
