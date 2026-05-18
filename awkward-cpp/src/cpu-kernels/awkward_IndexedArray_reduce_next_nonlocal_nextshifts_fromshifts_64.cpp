// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
  int64_t* nextshifts,
  const T* index,
  int64_t length,
  const int64_t* shifts) {
  int64_t nullsum = 0;
  int64_t k = 0;
  for (int64_t i = 0;  i < length;  i++) {
    if (index[i] >= 0) {
      nextshifts[k] = shifts[i] + nullsum;
      k++;
    }
    else {
      nullsum++;
    }
  }
  return success();
}

#define WRAPPER(SUFFIX, T) \
  ERROR awkward_IndexedArray##SUFFIX(int64_t* nextshifts, const T* index, int64_t length, const int64_t* shifts) { \
    return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64<T>(nextshifts, index, length, shifts); \
  }

WRAPPER(32_reduce_next_nonlocal_nextshifts_fromshifts_64, int32_t)
WRAPPER(U32_reduce_next_nonlocal_nextshifts_fromshifts_64, uint32_t)
WRAPPER(64_reduce_next_nonlocal_nextshifts_fromshifts_64, int64_t)
