// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)                                              \
  FILENAME_FOR_EXCEPTIONS_CUDA(                                     \
      "src/cuda-kernels/"                                           \
      "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64.up", \
      line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

template <typename T>
__global__ void
awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_filter_mask(
    int8_t* filtered_mask_k,
    int8_t* filtered_mask_nullsum,
    const T* index,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if (index[thread_id] >= 0) {
      filtered_mask_k[thread_id] = 1;
    } else {
      filtered_mask_nullsum[thread_id] = 1;
    }
  }
}

template <typename T>
__global__ void
awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_kernel(
    int64_t* nextshifts,
    const T* index,
    int64_t length,
    int64_t* prefixed_mask_k,
    int64_t* prefixed_mask_nullsum) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if (index[thread_id] >= 0) {
      nextshifts[prefixed_mask_k[thread_id] - 1] =
          prefixed_mask_nullsum[thread_id] - 1;
    }
  }
}

template <typename T>
ERROR
awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64(int64_t* nextshifts,
                                                        const T* index,
                                                        int64_t length) {
  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  int8_t* filtered_mask_k;
  int8_t* filtered_mask_nullsum;
  HANDLE_ERROR(cudaMalloc((void**)&filtered_mask_k, sizeof(int8_t) * length));
  HANDLE_ERROR(
      cudaMalloc((void**)&filtered_mask_nullsum, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_mask_k, 0, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_mask_nullsum, 0, sizeof(int8_t) * length));

  awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_filter_mask<<<
      blocks_per_grid,
      threads_per_block>>>(
      filtered_mask_k, filtered_mask_nullsum, index, length);

  int64_t* prefixed_mask_k;
  HANDLE_ERROR(cudaMalloc((void**)&prefixed_mask_k, sizeof(int64_t) * length));

  int64_t* prefixed_mask_nullsum;
  HANDLE_ERROR(
      cudaMalloc((void**)&prefixed_mask_nullsum, sizeof(int64_t) * length));

  exclusive_scan(prefixed_mask_k, filtered_mask_k, length);
  exclusive_scan(prefixed_mask_nullsum, filtered_mask_nullsum, length);

  awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_kernel<<<
      blocks_per_grid,
      threads_per_block>>>(
      nextshifts, index, length, prefixed_mask_k, prefixed_mask_nullsum);

  cudaDeviceSynchronize();

  return success();
}
ERROR
awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_64(int64_t* nextshifts,
                                                          const int32_t* index,
                                                          int64_t length) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64<int32_t>(
      nextshifts, index, length);
}
ERROR
awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_64(
    int64_t* nextshifts, const uint32_t* index, int64_t length) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64<uint32_t>(
      nextshifts, index, length);
}
ERROR
awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_64(int64_t* nextshifts,
                                                          const int64_t* index,
                                                          int64_t length) {
  return awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64<int64_t>(
      nextshifts, index, length);
}
