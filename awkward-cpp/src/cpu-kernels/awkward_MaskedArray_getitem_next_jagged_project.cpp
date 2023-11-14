// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_MaskedArray_getitem_next_jagged_project.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_MaskedArray_getitem_next_jagged_project(
  T* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0; i < length; ++i) {
    if (index[i] >= 0) {
      starts_out[k] = starts_in[i];
      stops_out[k] = stops_in[i];
      k++;
    }
  }
  return success();
}
ERROR awkward_MaskedArray32_getitem_next_jagged_project(
  int32_t* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<int32_t>(
    index,
    starts_in,
    stops_in,
    starts_out,
    stops_out,
    length);
}
ERROR awkward_MaskedArrayU32_getitem_next_jagged_project(
  uint32_t* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<uint32_t>(
    index,
    starts_in,
    stops_in,
    starts_out,
    stops_out,
    length);
}
ERROR awkward_MaskedArray64_getitem_next_jagged_project(
  int64_t* index,
  int64_t* starts_in,
  int64_t* stops_in,
  int64_t* starts_out,
  int64_t* stops_out,
  int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<int64_t>(
    index,
    starts_in,
    stops_in,
    starts_out,
    stops_out,
    length);
}
