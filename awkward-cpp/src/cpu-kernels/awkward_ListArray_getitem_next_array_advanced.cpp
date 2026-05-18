// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListArray_getitem_next_array_advanced.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_ListArray_getitem_next_array_advanced(
  T* tocarry,
  T* toadvanced,
  const C* fromstarts,
  const C* fromstops,
  const T* fromarray,
  const T* fromadvanced,
  int64_t lenstarts,
  int64_t lencontent) {
  for (int64_t i = 0;  i < lenstarts;  i++) {
    if (fromstops[i] < fromstarts[i]) {
      return failure("stops[i] < starts[i]", i, kSliceNone, FILENAME(__LINE__));
    }
    if ((fromstarts[i] != fromstops[i])  &&
        (fromstops[i] > lencontent)) {
      return failure("stops[i] > len(content)", i, kSliceNone, FILENAME(__LINE__));
    }
    int64_t length = fromstops[i] - fromstarts[i];
    int64_t regular_at = fromarray[fromadvanced[i]];
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      return failure("index out of range", i, fromarray[fromadvanced[i]], FILENAME(__LINE__));
    }
    tocarry[i] = fromstarts[i] + regular_at;
    toadvanced[i] = i;
  }
  return success();
}

#define WRAPPER(SUFFIX, C, T) \
  ERROR awkward_ListArray##SUFFIX(T* tocarry, T* toadvanced, const C* fromstarts, const C* fromstops, const T* fromarray, const T* fromadvanced, int64_t lenstarts, int64_t lencontent) { \
    return awkward_ListArray_getitem_next_array_advanced<C, T>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, lenstarts, lencontent); \
  }

WRAPPER(32_getitem_next_array_advanced_64, int32_t, int64_t)
WRAPPER(U32_getitem_next_array_advanced_64, uint32_t, int64_t)
WRAPPER(64_getitem_next_array_advanced_64, int64_t, int64_t)
