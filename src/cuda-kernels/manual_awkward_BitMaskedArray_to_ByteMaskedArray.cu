#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_BitMaskedArray_to_ByteMaskedArray.cu", line)

#include "awkward/kernels.h"
#include "awkward/cuda-utils.h"

__global__ void
cuda_BitMaskedArray_to_ByteMaskedArray(int8_t* tobytemask,
                                       const uint8_t* frombitmask,
                                       int64_t bitmasklength,
                                       bool validwhen,
                                       bool lsb_order) {
  int64_t threadx = blockIdx.x * blockDim.x + threadIdx.x;
  if (lsb_order) {
    if (threadx < bitmasklength) {
      auto byte = frombitmask[threadx];
      tobytemask[((threadx * 8) + 0)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 1)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 2)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 3)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 4)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 5)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 6)] = (byte & (uint8_t)(1)) != validwhen;
      byte >>= 1;
      tobytemask[((threadx * 8) + 7)] = (byte & (uint8_t)(1)) != validwhen;
    }

  } else {
    if (threadx < bitmasklength) {
      auto byte = frombitmask[threadx];
      tobytemask[((threadx * 8) + 0)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 1)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 2)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 3)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 4)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 5)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 6)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
      byte <<= 1;
      tobytemask[((threadx * 8) + 7)] =
          ((byte & (uint8_t)(128)) != 0) != validwhen;
    }
  }
}

ERROR
awkward_BitMaskedArray_to_ByteMaskedArray(int8_t* tobytemask,
                                          const uint8_t* frombitmask,
                                          int64_t bitmasklength,
                                          bool validwhen,
                                          bool lsb_order) {
  dim3 blocks_per_grid = blocks(bitmasklength);
  dim3 threads_per_block = threads(bitmasklength);

  cuda_BitMaskedArray_to_ByteMaskedArray<<<blocks_per_grid, threads_per_block>>>(
      tobytemask, frombitmask, bitmasklength, validwhen, lsb_order);
  return post_kernel_checks();
}
