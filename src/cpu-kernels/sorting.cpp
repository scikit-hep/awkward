// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cstring>

#include "awkward/cpu-kernels/sorting.h"

EXPORT_SYMBOL struct Error
  awkward_argsort_prepare_ranges(
    int64_t* toranges,
    const int64_t* parents,
    int64_t parentsoffset,
    int64_t length,
    int64_t outlength) {
  for (int64_t i = 0; i < length; i++) {
    toranges[i] = parents[i];
  }
  toranges[length] = outlength;

  return success();
}

template <typename OUT>
ERROR awkward_argsort(
  OUT* toptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = fromindex[i];
  }
  return success();
}
ERROR awkward_argsort_64(
  int64_t* toptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  int64_t length) {
  return awkward_argsort<int64_t>(
    toptr,
    fromindex,
    indexoffset,
    length);
}

template <typename T>
ERROR awkward_sort(
  T* toptr,
  const T* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    int64_t parent = parents[parentsoffset + i];
    int64_t start = starts[parent];

    toptr[i] = fromptr[fromindex[i] + indexoffset];
  }
  return success();
}
ERROR awkward_sort_bool(
  bool* toptr,
  const bool* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<bool>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_int8(
  int8_t* toptr,
  const int8_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<int8_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_uint8(
  uint8_t* toptr,
  const uint8_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<uint8_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_int16(
  int16_t* toptr,
  const int16_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<int16_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_uint16(
  uint16_t* toptr,
  const uint16_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<uint16_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_int32(
  int32_t* toptr,
  const int32_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<int32_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_uint32(
  uint32_t* toptr,
  const uint32_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<uint32_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_int64(
  int64_t* toptr,
  const int64_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<int64_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_uint64(
  uint64_t* toptr,
  const uint64_t* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<uint64_t>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_float32(
  float* toptr,
  const float* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<float>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}
ERROR awkward_sort_float64(
  double* toptr,
  const double* fromptr,
  const int64_t* fromindex,
  int64_t indexoffset,
  const int64_t* starts,
  const int64_t* parents,
  int64_t parentsoffset,
  int64_t length) {
  return awkward_sort<double>(
    toptr,
    fromptr,
    fromindex,
    indexoffset,
    starts,
    parents,
    parentsoffset,
    length);
}

ERROR awkward_listoffsetarray_local_preparenext_64(
  int64_t* outcarry,
  const int64_t* result,
  int64_t nextlen) {
  for(int64_t i = 0; i < nextlen; i++) {
    outcarry[i] = result[i];
  }
  return success();
}
