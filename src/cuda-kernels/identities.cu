// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/cuda_identities.h"
#include <stdio.h>

__global__
void cuda_Identities32_toIdentities64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length) {
  int64_t block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;
  if(thread_id < length) {
    toptr[thread_id] = (int64_t)(fromptr[thread_id]);
  }
}

ERROR awkward_cuda_Identities32_to_Identities64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  int64_t width) {

  dim3 blocks_per_grid;
  dim3 threads_per_block;

  if (length > 1024) {
    blocks_per_grid = dim3(ceil((length) / 1024.0), 1, 1);
    threads_per_block = dim3(1024, 1, 1);
  } else {
    blocks_per_grid = dim3(1, 1, 1);
    threads_per_block = dim3(length, 1, 1);
  }

  cuda_Identities32_toIdentities64<<<blocks_per_grid, threads_per_block>>>(
    toptr,
    fromptr,
    length * width);

  cudaDeviceSynchronize();

  return success();
}
