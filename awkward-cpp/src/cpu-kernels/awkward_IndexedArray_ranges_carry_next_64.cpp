// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_IndexedArray_ranges_carry_next_64.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_IndexedArray_ranges_carry_next_64(
  const T* index,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t* tocarry) {
  int64_t k = 0;
  int64_t stride = 0;
  for (int64_t i = 0; i < length; i++) {
    stride = fromstops[i] - fromstarts[i];
    for (int64_t j = 0; j < stride; j++) {
      if (!(index[fromstarts[i] + j] < 0)) {
        tocarry[k++] = index[fromstarts[i] + j];
      }
    }
  }

  return success();
}
ERROR awkward_IndexedArray32_ranges_carry_next_64(
  const int32_t* index,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t* tocarry) {
  return awkward_IndexedArray_ranges_carry_next_64<int32_t>(
    index,
    fromstarts,
    fromstops,
    length,
    tocarry);
}
ERROR awkward_IndexedArrayU32_ranges_carry_next_64(
  const uint32_t* index,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t* tocarry) {
  return awkward_IndexedArray_ranges_carry_next_64<uint32_t>(
    index,
    fromstarts,
    fromstops,
    length,
    tocarry);
}
ERROR awkward_IndexedArray64_ranges_carry_next_64(
  const int64_t* index,
  const int64_t* fromstarts,
  const int64_t* fromstops,
  int64_t length,
  int64_t* tocarry) {
  return awkward_IndexedArray_ranges_carry_next_64<int64_t>(
    index,
    fromstarts,
    fromstops,
    length,
    tocarry);
}
