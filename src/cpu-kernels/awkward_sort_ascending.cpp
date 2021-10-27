// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort_ascending.cpp", line)

#include "awkward/kernels.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

template <typename T>
bool sort_ascending(T l, T r)
{
  return !std::isnan(static_cast<double>(r)) && (std::isnan(static_cast<double>(l)) || l < r);
}

template <typename T>
ERROR awkward_sort_ascending(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), 0);

  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    auto start = std::next(index.begin(), offsets[i]);
    auto stop = std::next(index.begin(), offsets[i + 1]);
    std::stable_sort(start, stop, [&fromptr](int64_t i1, int64_t i2) {
      return sort_ascending<T>(fromptr[i1], fromptr[i2]);
    });
  }

  for (int64_t i = 0;  i < parentslength;  i++) {
    toptr[i] = fromptr[index[i]];
  }

  return success();
}
ERROR awkward_sort_ascending_bool(
  bool* toptr,
  const bool* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<bool>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<int8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<uint8_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<int16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<uint16_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<int32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<uint32_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<int64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<uint64_t>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_float32(
  float* toptr,
  const float* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<float>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}
ERROR awkward_sort_ascending_float64(
  double* toptr,
  const double* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t parentslength) {
  return awkward_sort_ascending<double>(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength);
}

template <>
bool sort_ascending(bool l, bool r)
{
  return l < r;
}
