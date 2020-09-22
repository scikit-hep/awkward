#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_NumpyArray_getitem_boolean_nonzero.cu", line)

#include "awkward/kernels/getitem.h"
#include "standard_parallel_algorithms.h"

__global__ void
awkward_NumpyArray_getitem_boolean_nonzero_filter_mask(
    const int8_t* fromptr,
    int64_t* filtered_index,
    int64_t stride,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length  && thread_id % stride == 0) {
    if (fromptr[thread_id] != 0) {
      filtered_index[thread_id] = 1;
    }
  }
}

template <typename T>
__global__ void
awkward_NumpyArray_getitem_boolean_nonzero_kernel(
    T* toptr,
    int64_t* prefixedsum_mask,
    const int8_t* fromptr,
    int64_t length,
    int64_t stride) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length && thread_id % stride == 0) {
    if(fromptr[thread_id] != 0) {
      toptr[prefixedsum_mask[thread_id] - 1] = thread_id;
    }
  }
}


template <typename T>
ERROR awkward_NumpyArray_getitem_boolean_nonzero(
    T* toptr,
    const int8_t* fromptr,
    int64_t length,
    int64_t stride) {

  int64_t* res_temp;
  int64_t* filtered_index;

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&filtered_index, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_index, 0, sizeof(int64_t) * length));


  awkward_NumpyArray_getitem_boolean_nonzero_filter_mask<<<
  blocks_per_grid,
  threads_per_block>>>(fromptr, filtered_index, stride, length);


  exclusive_scan<int64_t, int64_t>(res_temp, filtered_index, length);


  awkward_NumpyArray_getitem_boolean_nonzero_kernel<<<blocks_per_grid,
  threads_per_block>>>(
      toptr,
      res_temp,
      fromptr,
      length,
      stride);

  cudaDeviceSynchronize();

  return success();


}

ERROR awkward_NumpyArray_getitem_boolean_nonzero_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    int64_t stride) {
  return awkward_NumpyArray_getitem_boolean_nonzero<int64_t>(
      toptr,
      fromptr,
      length,
      stride);
}
