#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_Content_getitem_next_missing_jagged_getmaskstartstop.cu", line)

#include "awkward/kernels/getitem.h"
#include "standard_parallel_algorithms.h"

template <typename T>
__global__ void
awkward_MaskedArray_getitem_next_jagged_project_filter_mask(
    T* index,
    int64_t* filtered_index,
    int64_t length) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (index[thread_id] >= 0) {
      filtered_index[thread_id] = 1;
    }
  }
}

template <typename T>
__global__ void
awkward_MaskedArray_getitem_next_jagged_project_kernel(
    T* index,
    int64_t* prefixedsum_mask,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (index[thread_id] >= 0) {
      starts_out[prefixedsum_mask[thread_id] - 1] = starts_in[thread_id];
      stops_out[prefixedsum_mask[thread_id] - 1] = stops_in[thread_id];
    }
  }
}

template <typename T>
ERROR awkward_MaskedArray_getitem_next_jagged_project(
    T* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  int64_t* res_temp;
  int64_t* filtered_index;

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&filtered_index, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_index, 0, sizeof(int64_t) * length));


  awkward_MaskedArray_getitem_next_jagged_project_filter_mask<<<
  blocks_per_grid,
  threads_per_block>>>(index, filtered_index, length);


  exclusive_scan<int64_t, int64_t>(res_temp, filtered_index, length);


  awkward_MaskedArray_getitem_next_jagged_project_kernel<<<blocks_per_grid,
  threads_per_block>>>(
      index,
      res_temp,
      starts_in,
      stops_in,
      starts_out,
      stops_out,
      length);

  cudaDeviceSynchronize();

  return success();
}

ERROR awkward_MaskedArray32_getitem_next_jagged_project(
    int32_t* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<int32_t>(
      index,
      starts_in,
      stops_in,
      starts_out,
      stops_out,
      length);
}
ERROR awkward_MaskedArrayU32_getitem_next_jagged_project(
    uint32_t* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<uint32_t>(
      index,
      starts_in,
      stops_in,
      starts_out,
      stops_out,
      length);
}
ERROR awkward_MaskedArray64_getitem_next_jagged_project(
    int64_t* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length) {
  return awkward_MaskedArray_getitem_next_jagged_project<int64_t>(
      index,
      starts_in,
      stops_in,
      starts_out,
      stops_out,
      length);
}
