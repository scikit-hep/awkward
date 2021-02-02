// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_carry_SliceJagged64_nextcarry.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_carry_SliceJagged_nextcarry(
  int64_t* tocarry,
  const T* fromoffsets,
  const int64_t* fromcarry,
  int64_t carrylen) {
  int64_t k = 0;
  for (int64_t i = 0;  i < carrylen;  i++) {
    int64_t c = fromcarry[i];
    T start = fromoffsets[c];
    T stop = fromoffsets[c + 1];
    for (int64_t j = start;  j < stop;  j++) {
      tocarry[k] = j;
      k++;
    }
  }
  return success();
}

ERROR awkward_carry_SliceJagged64_nextcarry(
  int64_t* tocarry,
  const int64_t* fromoffsets,
  const int64_t* fromcarry,
  int64_t carrylen) {
  return awkward_carry_SliceJagged_nextcarry<int64_t>(tocarry,
                                                      fromoffsets,
                                                      fromcarry,
                                                      carrylen);
}
