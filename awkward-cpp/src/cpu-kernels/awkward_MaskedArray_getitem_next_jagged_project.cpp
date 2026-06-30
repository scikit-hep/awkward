// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_MaskedArray_getitem_next_jagged_project.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_MaskedArray_getitem_next_jagged_project(
  T* __restrict__ index,
  int64_t* __restrict__ starts_in,
  int64_t* __restrict__ stops_in,
  int64_t* __restrict__ starts_out,
  int64_t* __restrict__ stops_out,
  int64_t length) {
  int64_t k = 0;
  for (int64_t i = 0; i < length; ++i) {
    bool keep = (index[i] >= 0);
    if (keep) {
        starts_out[k] = starts_in[i];
        stops_out[k] = stops_in[i];
        k++;
    }
  }
  return success();
}

#define WRAPPER(FUNC, T) \
  ERROR FUNC(T* index, int64_t* starts_in, int64_t* stops_in, int64_t* starts_out, int64_t* stops_out, int64_t length) { \
    return awkward_MaskedArray_getitem_next_jagged_project<T>(index, starts_in, stops_in, starts_out, stops_out, length); \
  }

WRAPPER(awkward_MaskedArray32_getitem_next_jagged_project, int32_t)
WRAPPER(awkward_MaskedArrayU32_getitem_next_jagged_project, uint32_t)
WRAPPER(awkward_MaskedArray64_getitem_next_jagged_project, int64_t)
