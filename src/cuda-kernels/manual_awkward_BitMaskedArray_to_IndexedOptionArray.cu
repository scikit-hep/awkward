#define FILENAME(line)                                                    \
  FILENAME_FOR_EXCEPTIONS_CUDA(                                           \
      "src/cuda-kernels/awkward_BitMaskedArray_to_IndexedOptionArray.cu", \
      line)

#include "awkward/kernels.h"
#include "awkward-cuda/cuda-functions.h"

__global__ void
cuda_BitMaskedArray_to_IndexedOptionArray(int64_t* toindex,
                                          const uint8_t* frombitmask,
                                          int64_t bitmasklength,
                                          bool validwhen,
                                          bool lsb_order) {
  int64_t threadx = blockIdx.x * blockDim.x + threadIdx.x;
  if (lsb_order) {
    if (threadx < bitmasklength) {
      auto byte = frombitmask[threadx];
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 0)] = ((threadx * 8) + 0);

      } else {
        toindex[((threadx * 8) + 0)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 1)] = ((threadx * 8) + 1);

      } else {
        toindex[((threadx * 8) + 1)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 2)] = ((threadx * 8) + 2);

      } else {
        toindex[((threadx * 8) + 2)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 3)] = ((threadx * 8) + 3);

      } else {
        toindex[((threadx * 8) + 3)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 4)] = ((threadx * 8) + 4);

      } else {
        toindex[((threadx * 8) + 4)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 5)] = ((threadx * 8) + 5);

      } else {
        toindex[((threadx * 8) + 5)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 6)] = ((threadx * 8) + 6);

      } else {
        toindex[((threadx * 8) + 6)] = -1;
      }
      byte >>= 1;
      if (((byte & (uint8_t)(1)) == validwhen)) {
        toindex[((threadx * 8) + 7)] = ((threadx * 8) + 7);

      } else {
        toindex[((threadx * 8) + 7)] = -1;
      }
    }

  } else {
    if (threadx < bitmasklength) {
      auto byte = frombitmask[threadx];
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 0)] = ((threadx * 8) + 0);

      } else {
        toindex[((threadx * 8) + 0)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 1)] = ((threadx * 8) + 1);

      } else {
        toindex[((threadx * 8) + 1)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 2)] = ((threadx * 8) + 2);

      } else {
        toindex[((threadx * 8) + 2)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 3)] = ((threadx * 8) + 3);

      } else {
        toindex[((threadx * 8) + 3)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 4)] = ((threadx * 8) + 4);

      } else {
        toindex[((threadx * 8) + 4)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 5)] = ((threadx * 8) + 5);

      } else {
        toindex[((threadx * 8) + 5)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 6)] = ((threadx * 8) + 6);

      } else {
        toindex[((threadx * 8) + 6)] = -1;
      }
      byte <<= 1;
      if ((((byte & (uint8_t)(128)) != 0) == validwhen)) {
        toindex[((threadx * 8) + 7)] = ((threadx * 8) + 7);

      } else {
        toindex[((threadx * 8) + 7)] = -1;
      }
    }
  }
}

ERROR
awkward_BitMaskedArray_to_IndexedOptionArray64(int64_t* toindex,
                                               const uint8_t* frombitmask,
                                               int64_t bitmasklength,
                                               bool validwhen,
                                               bool lsb_order) {
  dim3 blocks_per_grid = blocks(bitmasklength);
  dim3 threads_per_block = threads(bitmasklength);

  cuda_BitMaskedArray_to_IndexedOptionArray<<<blocks_per_grid,
                                              threads_per_block>>>(
      toindex, frombitmask, bitmasklength, validwhen, lsb_order);
  return post_kernel_checks();
}
