// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
  int64_t* nextstarts,
  const int64_t* nextparents,
  int64_t nextlen) {
  int64_t lastnextparent = -1;
  for (int64_t i = 0;  i < nextlen;  i++) {
    if (nextparents[i] != lastnextparent) {
      nextstarts[nextparents[i]] = i;
    }
    lastnextparent = nextparents[i];
  }
  return success();
}
