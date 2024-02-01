// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort.cpp", line)

#include "awkward/kernels.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

template <typename T>
bool sort_order_ascending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l < r);
}

template <typename T>
bool sort_order_descending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l > r);
}

template <typename T>
ERROR awkward_sort(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), 0);

  if (ascending  &&  stable) {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(index.begin(), offsets[i]);
      auto stop = std::next(index.begin(), offsets[i + 1]);
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return sort_order_ascending<T>(fromptr[i1], fromptr[i2]);
      });
    }
  }
  else if (!ascending  &&  stable) {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(index.begin(), offsets[i]);
      auto stop = std::next(index.begin(), offsets[i + 1]);
      std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return sort_order_descending<T>(fromptr[i1], fromptr[i2]);
      });
    }
  }
  else if (ascending  &&  !stable) {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(index.begin(), offsets[i]);
      auto stop = std::next(index.begin(), offsets[i + 1]);
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return sort_order_ascending<T>(fromptr[i1], fromptr[i2]);
      });
    }
  }
  else {
    for (int64_t i = 0;  i < offsetslength - 1;  i++) {
      auto start = std::next(index.begin(), offsets[i]);
      auto stop = std::next(index.begin(), offsets[i + 1]);
      std::sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
        return sort_order_descending<T>(fromptr[i1], fromptr[i2]);
      });
    }
  }

  for (int64_t i = 0;  i < parentslength;  i++) {
    toptr[i] = fromptr[index[i]];
  }

  return success();
}
ERROR awkward_sort_bool(
  bool* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_float32(
  float* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}
ERROR awkward_sort_float64(
  double* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength,
  bool ascending,
  bool stable) {
  return awkward_sort<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable);
}

template <>
bool sort_order_ascending(bool l, bool r)
{
  return l < r;
}

template <>
bool sort_order_descending(bool l, bool r)
{
  return l > r;
}
