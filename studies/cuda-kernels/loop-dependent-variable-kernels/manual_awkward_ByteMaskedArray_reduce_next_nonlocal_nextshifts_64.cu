// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line)                                                  \
  FILENAME_FOR_EXCEPTIONS_CUDA(                                         \
      "src/cpu-kernels/"                                                \
      "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64.cpp", \
      line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

__global__ void
awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_filter_mask(
    int8_t* filtered_mask_k,
    int8_t* filtered_mask_nullsum,
    const int8_t* mask,
    bool valid_when,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if ((mask[thread_id] != 0) == (valid_when != 0)) {
      filtered_mask_k[thread_id] = 1;
    } else {
      filtered_mask_nullsum[thread_id] = 1;
    }
  }
}
__global__ void
awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_kernel(
    int64_t* nextshifts,
    const int8_t* mask,
    bool valid_when,
    int64_t length,
    int64_t* prefixed_mask_k,
    int64_t* prefixed_mask_nullsum) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if ((mask[thread_id] != 0) == (valid_when != 0)) {
      nextshifts[prefixed_mask_k[thread_id] - 1] =
          prefixed_mask_nullsum[thread_id] - 1;
    }
  }
}

ERROR
awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(int64_t* nextshifts,
                                                           const int8_t* mask,
                                                           int64_t length,
                                                           bool valid_when) {
  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  int8_t* filtered_mask_k;
  int8_t* filtered_mask_nullsum;
  HANDLE_ERROR(cudaMalloc((void**)&filtered_mask_k, sizeof(int8_t) * length));
  HANDLE_ERROR(
      cudaMalloc((void**)&filtered_mask_nullsum, sizeof(int8_t) * length));
  HANDLE_ERROR(cudaMemset(filtered_mask_k, length, 0));
  HANDLE_ERROR(cudaMemset(filtered_mask_nullsum, length, 0));

  awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_filter_mask<<<
      blocks_per_grid,
      threads_per_block>>>(
      filtered_mask_k, filtered_mask_nullsum, mask, valid_when, length);

  int64_t* prefixed_mask_k;
  HANDLE_ERROR(cudaMalloc((void**)&prefixed_mask_k, sizeof(int64_t) * length));

  int64_t* prefixed_mask_nullsum;
  HANDLE_ERROR(
      cudaMalloc((void**)&prefixed_mask_nullsum, sizeof(int64_t) * length));

  exclusive_scan(prefixed_mask_k, filtered_mask_k, length);
  exclusive_scan(prefixed_mask_nullsum, filtered_mask_nullsum, length);

  awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_kernel<<<
      blocks_per_grid,
      threads_per_block>>>(
      nextshifts, mask, valid_when, length, prefixed_mask_k, prefixed_mask_nullsum);

  cudaDeviceSynchronize();

  return success();
}
