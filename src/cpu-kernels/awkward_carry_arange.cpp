// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_carry_arange.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_carry_arange(
  T* toptr,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toptr[i] = i;
  }
  return success();
}
ERROR awkward_carry_arange32(
  int32_t* toptr,
  int64_t length) {
  return awkward_carry_arange<int32_t>(
    toptr,
    length);
}
ERROR awkward_carry_arangeU32(
  uint32_t* toptr,
  int64_t length) {
  return awkward_carry_arange<uint32_t>(
    toptr,
    length);
}
ERROR awkward_carry_arange64(
  int64_t* toptr,
  int64_t length) {
  return awkward_carry_arange<int64_t>(
    toptr,
    length);
}
