// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_IndexedArray_reduce_next_64.cu", line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

template <typename T>
__global__ void
awkward_IndexedArray_reduce_next_64_filter_mask(int8_t* filtered_mask,
                                                const T* index,
                                                int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if (index[thread_id] >= 0) {
      filtered_mask[thread_id] = 1;
    }
  }
}

template <typename T>
__global__ void
awkward_IndexedArray_reduce_next_64_kernel(int64_t* nextcarry,
                                           int64_t* nextparents,
                                           int64_t* outindex,
                                           const T* index,
                                           const int64_t* parents,
                                           int64_t length,
                                           int64_t* prefixed_mask) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if (index[thread_id] >= 0) {
      nextcarry[prefixed_mask[thread_id] - 1] = index[thread_id];
      nextparents[prefixed_mask[thread_id] - 1] = parents[thread_id];
      outindex[thread_id] = prefixed_mask[thread_id] - 1;
    } else {
      outindex[thread_id] = -1;
    }
  }
}

template <typename T>
ERROR
awkward_IndexedArray_reduce_next_64(int64_t* nextcarry,
                                    int64_t* nextparents,
                                    int64_t* outindex,
                                    const T* index,
                                    const int64_t* parents,
                                    int64_t length) {
  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  int8_t* filtered_mask;
  HANDLE_ERROR(cudaMalloc((void**)&filtered_mask, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMemset(
      filtered_mask, 0, sizeof(int8_t) * length));

  awkward_IndexedArray_reduce_next_64_filter_mask<<<blocks_per_grid,
                                                    threads_per_block>>>(
      filtered_mask, index, length);

  int64_t* prefixed_mask;
  HANDLE_ERROR(cudaMalloc((void**)&prefixed_mask, sizeof(int64_t) * length));

  exclusive_scan(prefixed_mask, filtered_mask, length);

  awkward_IndexedArray_reduce_next_64_kernel<<<blocks_per_grid,
                                               threads_per_block>>>(
      nextcarry, nextparents, outindex, index, parents, length, prefixed_mask);

  cudaDeviceSynchronize();

  return success();
}
ERROR
awkward_IndexedArray32_reduce_next_64(int64_t* nextcarry,
                                      int64_t* nextparents,
                                      int64_t* outindex,
                                      const int32_t* index,
                                      int64_t* parents,
                                      int64_t length) {
  return awkward_IndexedArray_reduce_next_64<int32_t>(
      nextcarry, nextparents, outindex, index, parents, length);
}
ERROR
awkward_IndexedArrayU32_reduce_next_64(int64_t* nextcarry,
                                       int64_t* nextparents,
                                       int64_t* outindex,
                                       const uint32_t* index,
                                       int64_t* parents,
                                       int64_t length) {
  return awkward_IndexedArray_reduce_next_64<uint32_t>(
      nextcarry, nextparents, outindex, index, parents, length);
}
ERROR
awkward_IndexedArray64_reduce_next_64(int64_t* nextcarry,
                                      int64_t* nextparents,
                                      int64_t* outindex,
                                      const int64_t* index,
                                      int64_t* parents,
                                      int64_t length) {
  return awkward_IndexedArray_reduce_next_64<int64_t>(
      nextcarry, nextparents, outindex, index, parents, length);
}
