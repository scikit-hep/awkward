#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_IndexedArray_getitem_adjust_outindex.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

__global__ void
awkward_IndexedArray_getitem_adjust_outindex_64_filter_k_and_mask(
    int64_t* fromindex,
    int8_t* filtered_k,
    int8_t* tomask,
    int64_t length) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    tomask[thread_id] = (fromindex[thread_id] < 0);
    if(fromindex[thread_id] < 0) {
      filtered_k[thread_id] = 1;
    }
    else if(thread_id < nonzerolength)
  }
}


__global__ void
awkward_IndexedArray_getitem_adjust_outindex_kernel(
    int64_t* prefixed_index,
    int64_t* index_in,
    int64_t* offsets_in,
    int64_t* mask_out,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    int64_t pre_in = prefixed_index[thread_id] - 1;
    starts_out[thread_id] = offsets_in[pre_in];

    if (index_in[thread_id] < 0) {
      mask_out[thread_id] = -1;
      stops_out[thread_id] = offsets_in[pre_in];
    } else {
      mask_out[thread_id] = thread_id;
      stops_out[thread_id] = offsets_in[pre_in + 1];
    }
  }
}

ERROR
awkward_IndexedArray_getitem_adjust_outindex_64(
    int8_t* tomask,
    int64_t* toindex,
    int64_t* tononzero,
    const int64_t* fromindex,
    int64_t fromindexlength,
    const int64_t* nonzero,
    int64_t nonzerolength) {

  int64_t* res_temp;
  int8_t* filtered_j;
  int8_t* filtered_k;

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&filtered_index, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemcpy(
      filtered_index, index_in, sizeof(int64_t) * length, cudaMemcpyDeviceToDevice));

  awkward_IndexedArray_getitem_adjust_outindex_64_filter_j<<<
  blocks_per_grid,
  threads_per_block>>>(nonzero, filtered_j, nonzerolength);


  awkward_IndexedArray_getitem_adjust_outindex_64_filter_k_and_mask<<<
  blocks_per_grid,
  threads_per_block>>>(fromindex, filtered_k, tomask, fromindexlength);


  exclusive_scan<int64_t, int64_t>(res_temp, filtered_index, length);


  awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel<<<blocks_per_grid,
  threads_per_block>>>(
      res_temp, index_in, offsets_in, mask_out, starts_out, stops_out, length);

  cudaDeviceSynchronize();

  return success();
}

