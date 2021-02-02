// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_carry_SliceJagged64_offsets.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_carry_SliceJagged_offsets(
  T* tooffsets,
  const T* fromoffsets,
  const int64_t* fromcarry,
  int64_t carrylen) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < carrylen;  i++) {
    int64_t c = fromcarry[i];
    T count = fromoffsets[c + 1] - fromoffsets[c];
    tooffsets[i + 1] = tooffsets[i] + count;
  }
  return success();
}

ERROR awkward_carry_SliceJagged64_offsets(
  int64_t* tooffsets,
  const int64_t* fromoffsets,
  const int64_t* fromcarry,
  int64_t carrylen) {
  return awkward_carry_SliceJagged_offsets<int64_t>(tooffsets,
                                                    fromoffsets,
                                                    fromcarry,
                                                    carrylen);
}
