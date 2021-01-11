// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_Index_to_Index64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_Index_to_Index64(
  int64_t* toptr,
  const T* fromptr,
  int64_t length
) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i]= (int64_t)fromptr[i];
  }
  return success();
}
ERROR awkward_Index8_to_Index64(
  int64_t* toptr,
  const int8_t* fromptr,
  int64_t length) {
  return awkward_Index_to_Index64<int8_t>(toptr, fromptr, length);
}
ERROR awkward_IndexU8_to_Index64(
  int64_t* toptr,
  const uint8_t* fromptr,
  int64_t length) {
  return awkward_Index_to_Index64<uint8_t>(toptr, fromptr, length);
}
ERROR awkward_Index32_to_Index64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length) {
  return awkward_Index_to_Index64<int32_t>(toptr, fromptr, length);
}
ERROR awkward_IndexU32_to_Index64(
  int64_t* toptr,
  const uint32_t* fromptr,
  int64_t length) {
  return awkward_Index_to_Index64<uint32_t>(toptr, fromptr, length);
}
