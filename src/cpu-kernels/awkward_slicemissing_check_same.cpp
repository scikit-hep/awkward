// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_slicemissing_check_same.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_slicemissing_check_same(
  bool* same,
  const int8_t* bytemask,
  const int64_t* missingindex,
  int64_t length) {
  *same = true;
  for (int64_t i = 0;  i < length;  i++) {
    bool left = (bytemask[i] != 0);
    bool right = (missingindex[i] < 0);
    if (left != right) {
      *same = false;
      return success();
    }
  }
  return success();
}
