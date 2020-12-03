// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_zeroparents_64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_content_reduce_zeroparents_64(
  int64_t* toparents,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toparents[i] = 0;
  }
  return success();
}
