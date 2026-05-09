// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_reduce_next_64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_IndexedArray_reduce_next_64(
  int64_t* nextcarry,
  int64_t* nextoffsets,   // length outlength + 1
  int64_t* outindex,
  const T* index,
  const int64_t* offsets,
  int64_t outlength) {
  int64_t k = 0;
  nextoffsets[0] = 0;
  for (int64_t bin = 0; bin < outlength; bin++) {
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      if (index[i] >= 0) {
        nextcarry[k] = index[i];
        outindex[i] = k;
        k++;
      }
      else {
        outindex[i] = -1;
      }
    }
    nextoffsets[bin + 1] = k;
  }
  return success();
}

#define WRAPPER(SUFFIX, T) \
  ERROR awkward_IndexedArray##SUFFIX(int64_t* nextcarry, int64_t* nextoffsets, int64_t* outindex, const T* index, const int64_t* offsets, int64_t outlength) { \
    return awkward_IndexedArray_reduce_next_64<T>(nextcarry, nextoffsets, outindex, index, offsets, outlength); \
  }

WRAPPER(32_reduce_next_64, int32_t)
WRAPPER(U32_reduce_next_64, uint32_t)
WRAPPER(64_reduce_next_64, int64_t)
