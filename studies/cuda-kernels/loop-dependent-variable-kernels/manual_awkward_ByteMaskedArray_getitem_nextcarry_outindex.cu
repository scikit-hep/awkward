#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_ByteMaskedArray_getitem_nextcarry_outindex.cu", line)


#include "standard_parallel_algorithms.h"
#include "awkward/kernels/getitem.h"

__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_outindex_filter_mask(int8_t* mask,
                                                               bool validwhen,
                                                               int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if ((mask[thread_id] != 0) == validwhen) {
      mask[thread_id] = 1;
    }
  }
}

__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel(int64_t* prefixed_mask,
                                                          int64_t* to_carry,
                                                          int64_t* outindex,
                                                          int8_t* mask,
                                                          int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (mask[thread_id] != 0) {
      to_carry[prefixed_mask[thread_id] - 1] = thread_id;
      outindex[thread_id] = prefixed_mask[thread_id] - 1;
    } else {
      outindex[thread_id] = -1;
    }
  }
}


ERROR
awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    int64_t* outindex,
    const int8_t* mask,
    int64_t length,
    bool validwhen) {
  int64_t* res_temp;
  int8_t* filtered_mask;

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMemcpy(
      filtered_mask, mask, sizeof(int8_t) * length, cudaMemcpyDeviceToDevice));

  awkward_ByteMaskedArray_getitem_nextcarry_outindex_filter_mask<<<blocks_per_grid, threads_per_block>>>(
      filtered_mask, validwhen, length);

  exclusive_scan<int64_t, int8_t>(res_temp, filtered_mask, length);

  awkward_ByteMaskedArray_getitem_nextcarry_outindex_kernel<<<blocks_per_grid, threads_per_block>>>(
      res_temp, tocarry, outindex, filtered_mask, length);

  cudaDeviceSynchronize();

  return success();
}
