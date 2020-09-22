#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_IndexedOptionArray_rpad_and_clip_mask_axis1.cu", line)

#include "awkward/kernels/operations.h"
#include "standard_parallel_algorithms.h"

__global__ void
awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_filter_mask(
    const int8_t* frommask,
    int8_t* filtered_mask,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (!frommask[thread_id]) {
      filtered_mask[thread_id] = 1;
    }
  }
}

template <typename T>
__global__ void
awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_kernel(
    T* toindex,
    const int8_t* frommask,
    int64_t* prefixedsum_mask,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (frommask[thread_id]) {
      toindex[thread_id] = -1;
    }
    else {
      toindex[thread_id] = prefixedsum_mask[thread_id] - 1;
    }
  }
}


template <typename T>
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1(
    T* toindex,
    const int8_t* frommask,
    int64_t length) {

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  int8_t* filtered_mask;
  int64_t* res_temp;

  HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_mask, 0, length));

  awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_filter_mask<<<blocks_per_grid, threads_per_block>>>(
      frommask,
      filtered_mask,
      length);

  exclusive_scan(res_temp, filtered_mask, length);

  awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_kernel<<<blocks_per_grid, threads_per_block>>>(
      toindex,
      frommask,
      res_temp,
      length);

  cudaDeviceSynchronize();

  return success();
}
ERROR awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
    int64_t* toindex,
    const int8_t* frommask,
    int64_t length) {
  return awkward_IndexedOptionArray_rpad_and_clip_mask_axis1<int64_t>(
      toindex,
      frommask,
      length);
}
