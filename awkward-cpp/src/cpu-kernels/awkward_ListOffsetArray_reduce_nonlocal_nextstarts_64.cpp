// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
//
// DEPRECATED — nextparents → nextstarts derivation.
//
// With the offsets-pipeline migration, `nextstarts` is simply
// `nextoffsets[:-1]` and is derived in Python without a kernel call. This
// function is preserved for ABI compatibility; remove once all callers have
// migrated.

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
