// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/manual_awkward_ListArray_num.cu", line)

// #include "awkward/kernels.h"
#include <cstdint>
__global__ void
cuda_BitMaskedArray_to_ByteMaskedArray(int8_t* tobytemask,
  const uint8_t* frombitmask,
  int64_t bitmasklength,
  bool validwhen,
  bool lsb_order,
  int8_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < bitmasklength) {
    if (lsb_order) {
      uint8_t byte = frombitmask[thread_id];
      tobytemask[thread_id*8 + 0] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 1] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 2] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 3] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 4] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 5] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 6] = ((byte & ((uint8_t)1)) != validwhen);
      byte >>= 1;
      tobytemask[thread_id*8 + 7] = ((byte & ((uint8_t)1)) != validwhen);
  }
  else {
      uint8_t byte = frombitmask[thread_id];
      tobytemask[thread_id*8 + 0] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 1] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 2] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 3] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 4] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 5] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 6] = (((byte & ((uint8_t)128)) != 0) != validwhen);
      byte <<= 1;
      tobytemask[thread_id*8 + 7] = (((byte & ((uint8_t)128)) != 0) != validwhen);
  }
}
}