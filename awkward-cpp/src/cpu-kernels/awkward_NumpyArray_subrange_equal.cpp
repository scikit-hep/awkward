// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_subrange_equal.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_subrange_equal(
  T* tmpptr,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  bool* toequal) {

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

#define WRAPPER(SUFFIX, T) \
  ERROR awkward_NumpyArray_subrange_equal_##SUFFIX(T* tmpptr, const int64_t* fromstarts, const int64_t* fromstops, int64_t length, bool* toequal) { \
    return awkward_NumpyArray_subrange_equal<T>(tmpptr, fromstarts, fromstops, length, toequal); \
  }

WRAPPER(int8, int8_t)
WRAPPER(uint8, uint8_t)
WRAPPER(int16, int16_t)
WRAPPER(uint16, uint16_t)
WRAPPER(int32, int32_t)
WRAPPER(uint32, uint32_t)
WRAPPER(int64, int64_t)
WRAPPER(uint64, uint64_t)
WRAPPER(float32, float)
WRAPPER(float64, double)
