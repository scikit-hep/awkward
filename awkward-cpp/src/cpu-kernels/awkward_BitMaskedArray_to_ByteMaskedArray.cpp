// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_BitMaskedArray_to_ByteMaskedArray.cpp", line)

#include "awkward/kernels.h"

ERROR awkward_BitMaskedArray_to_ByteMaskedArray(
  int8_t* tobytemask,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order) {
  if (lsb_order) {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      tobytemask[i*8 + 0] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 1] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 2] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 3] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 4] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 5] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 6] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[i*8 + 7] = ((byte & ((uint8_t)1)) != validwhen);
    }
  }
  else {
    for (int64_t i = 0;  i < bitmasklength;  i++) {
      uint8_t byte = frombitmask[i];
      tobytemask[i*8 + 0] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 1] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 2] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 3] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 4] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 5] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 6] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[i*8 + 7] = (((byte & ((uint8_t)128)) != 0) != validwhen);
    }
  }
  return success();
}
