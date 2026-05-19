// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_ranges.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_unique_ranges(
  T* toptr,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  int64_t* tooffsets) {
  int64_t m = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    tooffsets[i] = m;
    toptr[m++] = toptr[fromoffsets[i]];
    for (int64_t k = fromoffsets[i]; k < fromoffsets[i + 1]; k++) {
      if (toptr[m - 1] != toptr[k]) {
        toptr[m++] = toptr[k];
      }
    }
  }
  tooffsets[offsetslength - 1] = m;

  return success();
}

#define WRAPPER(SUFFIX, T) \
  ERROR awkward_unique_ranges_##SUFFIX(T* toptr, const int64_t* fromoffsets, int64_t offsetslength, int64_t* tooffsets) { \
    return awkward_unique_ranges<T>(toptr, fromoffsets, offsetslength, tooffsets); \
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
