// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_ListOffsetArray_local_preparenext_64.cpp", line)

#include <algorithm>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"

ERROR awkward_ListOffsetArray_local_preparenext_64(
  int64_t* tocarry,
  const int64_t* fromindex,
  int64_t length) {
  std::vector<int64_t> result(length);
  std::iota(result.begin(), result.end(), 0);
  std::sort(result.begin(), result.end(),
    [&fromindex](int64_t i1, int64_t i2) {
      return fromindex[i1] < fromindex[i2];
    });

  for(int64_t i = 0; i < length; i++) {
    tocarry[i] = result[i];
  }
  return success();
}
