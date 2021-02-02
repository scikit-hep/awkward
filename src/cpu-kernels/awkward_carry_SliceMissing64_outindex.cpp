// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_carry_SliceMissing64_outindex.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_carry_SliceJagged_nextcarry(
  T* toindex,
  const T* fromindex,
  int64_t length) {
  int64_t k = 0;
  int64_t j = 0;
  for (int64_t i = 0;  i < length;  i++) {
    T index = fromindex[i];
    if (index < 0) {
      toindex[i] = -1;
    }
    else {
      toindex[i] = j;
      j++;
    }
  }
  return success();
}

ERROR awkward_carry_SliceMissing64_outindex(
  int64_t* toindex,
  const int64_t* fromindex,
  int64_t length) {
  return awkward_carry_SliceJagged_nextcarry<int64_t>(toindex,
                                                      fromindex,
                                                      length);
}
