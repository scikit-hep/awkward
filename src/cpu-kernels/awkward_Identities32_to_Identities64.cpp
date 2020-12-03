// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Identities32_to_Identities64.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_Identities32_to_Identities64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  int64_t width) {
  for (int64_t i = 0;  i < length*width;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
