// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_BitMaskedArray_to_IndexedOptionArray.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_BitMaskedArray_to_IndexedOptionArray(
  T* toindex,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order) {
  if (lsb_order) {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 0] = i*8 + 0;
      }
      else {
        toindex[i*8 + 0] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 1] = i*8 + 1;
      }
      else {
        toindex[i*8 + 1] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 2] = i*8 + 2;
      }
      else {
        toindex[i*8 + 2] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 3] = i*8 + 3;
      }
      else {
        toindex[i*8 + 3] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 4] = i*8 + 4;
      }
      else {
        toindex[i*8 + 4] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 5] = i*8 + 5;
      }
      else {
        toindex[i*8 + 5] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 6] = i*8 + 6;
      }
      else {
        toindex[i*8 + 6] = -1;
      }
      byte >>= 1;
      if ((byte & ((uint8_t)1)) == validwhen) {
        toindex[i*8 + 7] = i*8 + 7;
      }
      else {
        toindex[i*8 + 7] = -1;
      }
    }
  }
  else {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 0] = i*8 + 0;
      }
      else {
        toindex[i*8 + 0] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 1] = i*8 + 1;
      }
      else {
        toindex[i*8 + 1] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 2] = i*8 + 2;
      }
      else {
        toindex[i*8 + 2] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 3] = i*8 + 3;
      }
      else {
        toindex[i*8 + 3] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 4] = i*8 + 4;
      }
      else {
        toindex[i*8 + 4] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 5] = i*8 + 5;
      }
      else {
        toindex[i*8 + 5] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 6] = i*8 + 6;
      }
      else {
        toindex[i*8 + 6] = -1;
      }
      byte <<= 1;
      if (((byte & ((uint8_t)128)) != 0) == validwhen) {
        toindex[i*8 + 7] = i*8 + 7;
      }
      else {
        toindex[i*8 + 7] = -1;
      }
    }
  }
  return success();
}
ERROR awkward_BitMaskedArray_to_IndexedOptionArray64(
  int64_t* toindex,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order) {
  return awkward_BitMaskedArray_to_IndexedOptionArray<int64_t>(
    toindex,
    frombitmask,
    bitmasklength,
    validwhen,
    lsb_order);
}
