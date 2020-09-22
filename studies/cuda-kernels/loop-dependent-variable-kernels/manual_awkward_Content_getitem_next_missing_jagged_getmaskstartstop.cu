#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_Content_getitem_next_missing_jagged_getmaskstartstop.cu", line)

#include "awkward/kernels/getitem.h"
#include "standard_parallel_algorithms.h"

__global__ void
awkward_Content_getitem_next_missing_jagged_getmaskstartstop_filter_mask(
    int64_t* index_in,
    int64_t* filtered_index,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (index_in[thread_id] >= 0) {
      filtered_index[thread_id] = 1;
    }
  }
}

__global__ void
awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel(
    int64_t* prefixed_index,
    int64_t* index_in,
    int64_t* offsets_in,
    int64_t* mask_out,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

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
awkward_Content_getitem_next_missing_jagged_getmaskstartstop(int64_t* index_in,
                                                             int64_t* offsets_in,
                                                             int64_t* mask_out,
                                                             int64_t* starts_out,
                                                             int64_t* stops_out,
                                                             int64_t length) {
  int64_t* res_temp;
  int64_t* filtered_index;
  int64_t* h_mask = new int64_t[length];

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&filtered_index, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemcpy(
      filtered_index, index_in, sizeof(int64_t) * length, cudaMemcpyDeviceToDevice));

  awkward_Content_getitem_next_missing_jagged_getmaskstartstop_filter_mask<<<
      blocks_per_grid,
      threads_per_block>>>(index_in, filtered_index, length);


  exclusive_scan<int64_t, int64_t>(res_temp, filtered_index, length);


  awkward_Content_getitem_next_missing_jagged_getmaskstartstop_kernel<<<blocks_per_grid,
                                                                        threads_per_block>>>(
      res_temp, index_in, offsets_in, mask_out, starts_out, stops_out, length);

  cudaDeviceSynchronize();

  return success();
}

