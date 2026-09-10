// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_subrange_equal.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_subrange_equal(
  T* __restrict__ tmpptr,
  const int64_t* __restrict__ fromstarts,
  const int64_t* __restrict__ fromstops,
  int64_t length,
  bool* __restrict__ toequal) {

  bool differ = true;
  int64_t leftlen;
  int64_t rightlen;

  for (int64_t i = 0;  i < length - 1;  i++) {
    leftlen = fromstops[i] - fromstarts[i];
    for (int64_t ii = i + 1; ii < length - 1;  ii++) {
      rightlen = fromstops[ii] - fromstarts[ii];
      if (leftlen == rightlen) {
        differ = false;
        for (int64_t j = 0; j < leftlen; j++) {
          if (tmpptr[fromstarts[i] + j] != tmpptr[fromstarts[ii] + j]) {
            differ = true;
            break;
          }
        }
      }
    }
  }

  *toequal = !differ;

  return success();
}

#define WRAPPER(FUNC, T) \
  ERROR FUNC(T* tmpptr, const int64_t* fromstarts, const int64_t* fromstops, int64_t length, bool* toequal) { \
    return awkward_NumpyArray_subrange_equal<T>(tmpptr, fromstarts, fromstops, length, toequal); \
  }

WRAPPER(awkward_NumpyArray_subrange_equal_int8, int8_t)
WRAPPER(awkward_NumpyArray_subrange_equal_uint8, uint8_t)
WRAPPER(awkward_NumpyArray_subrange_equal_int16, int16_t)
WRAPPER(awkward_NumpyArray_subrange_equal_uint16, uint16_t)
WRAPPER(awkward_NumpyArray_subrange_equal_int32, int32_t)
WRAPPER(awkward_NumpyArray_subrange_equal_uint32, uint32_t)
WRAPPER(awkward_NumpyArray_subrange_equal_int64, int64_t)
WRAPPER(awkward_NumpyArray_subrange_equal_uint64, uint64_t)
WRAPPER(awkward_NumpyArray_subrange_equal_float32, float)
WRAPPER(awkward_NumpyArray_subrange_equal_float64, double)
