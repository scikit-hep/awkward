// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_index_carry_nocheck.cpp", line)

#include "awkward/kernels.h"

template <typename C, typename T>
ERROR awkward_index_carry_nocheck(
  C* toindex,
  const C* fromindex,
  const T* carry,
  int64_t length) {
  for (int64_t i = 0;  i < length;  i++) {
    toindex[i] = fromindex[(size_t)(carry[i])];
  }
  return success();
}
ERROR awkward_Index8_carry_nocheck_64(
  int8_t* toindex,
  const int8_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<int8_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_IndexU8_carry_nocheck_64(
  uint8_t* toindex,
  const uint8_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<uint8_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_Index32_carry_nocheck_64(
  int32_t* toindex,
  const int32_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<int32_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_IndexU32_carry_nocheck_64(
  uint32_t* toindex,
  const uint32_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<uint32_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
ERROR awkward_Index64_carry_nocheck_64(
  int64_t* toindex,
  const int64_t* fromindex,
  const int64_t* carry,
  int64_t length) {
  return awkward_index_carry_nocheck<int64_t, int64_t>(
    toindex,
    fromindex,
    carry,
    length);
}
