// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_new_Identities.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_new_Identities(
  T* toptr,
  int64_t length) {
  for (T i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_new_Identities32(
  int32_t* toptr,
  int64_t length) {
  return awkward_new_Identities<int32_t>(
    toptr,
    length);
}
ERROR awkward_new_Identities64(
  int64_t* toptr,
  int64_t length) {
  return awkward_new_Identities<int64_t>(
    toptr,
    length);
}
