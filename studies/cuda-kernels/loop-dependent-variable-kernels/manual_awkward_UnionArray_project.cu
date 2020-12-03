#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_UnionArray_project.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

template <typename C>
__global__ void
awkward_UnionArray_project_filter_mask(
    const C* fromtags,
    int64_t which,
    int8_t* filtered_mask,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(thread_id < length) {
    if (fromtags[thread_id] == which) {
      filtered_mask[thread_id] = 1;
    }
  }
}

template <typename T, typename C, typename I>
__global__ void
awkward_UnionArray_project_kernel(
    int64_t* prefixedsum_mask,
    T* tocarry,
    const C* fromtags,
    const I* fromindex,
    int64_t length,
    int64_t which) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    if (fromtags[thread_id] == which) {
      tocarry[prefixedsum_mask[thread_id] - 1] = fromindex[thread_id];
    }
  }
}

template <typename T, typename C, typename I>
ERROR awkward_UnionArray_project(
    int64_t* lenout,
    T* tocarry,
    const C* fromtags,
    const I* fromindex,
    int64_t length,
    int64_t which) {

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  int8_t* filtered_mask;
  int64_t* res_temp;

  HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMalloc((void**)&res_temp, sizeof(int64_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_mask, 0, sizeof(int8_t) * length));

  awkward_UnionArray_project_filter_mask<C><<<blocks_per_grid, threads_per_block>>>(
      fromtags,
      which,
      filtered_mask,
      length);

  exclusive_scan(res_temp, filtered_mask, length);

  HANDLE_ERROR(cudaMemcpy(lenout, res_temp + (length - 1), sizeof(int64_t), cudaMemcpyDeviceToDevice));

  awkward_UnionArray_project_kernel<T, C, I><<<blocks_per_grid, threads_per_block>>>(
      res_temp,
      tocarry,
      fromtags,
      fromindex,
      length,
      which);

  cudaDeviceSynchronize();

  return success();
}


ERROR awkward_UnionArray8_32_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t length,
    int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, int32_t>(
      lenout,
      tocarry,
      fromtags,
      fromindex,
      length,
      which);
}

ERROR awkward_UnionArray8_U32_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t length,
    int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, uint32_t>(
      lenout,
      tocarry,
      fromtags,
      fromindex,
      length,
      which);
}
ERROR awkward_UnionArray8_64_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t length,
    int64_t which) {
  return awkward_UnionArray_project<int64_t, int8_t, int64_t>(
      lenout,
      tocarry,
      fromtags,
      fromindex,
      length,
      which);
}
