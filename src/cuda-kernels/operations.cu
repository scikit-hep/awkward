// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/cuda_operations.h"
#include <stdio.h>


template <typename T, typename C>
__global__
void cuda_ListArray_num(
  C *tonum,
  const T *fromstarts,
  int64_t startsoffset,
  const T *fromstops,
  int64_t stopsoffset
) {
  int64_t block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;
  int64_t start = fromstarts[startsoffset + thread_id];
  int64_t stop = fromstops[stopsoffset + thread_id];
  tonum[thread_id] = (C) (stop - start);
}


Error
awkward_cuda_ListArray32_num_64(
  int64_t* tonum,
  const int32_t* fromstarts,
  int64_t startsoffset,
  const int32_t* fromstops,
  int64_t stopsoffset,
  int64_t length) {

  dim3 blocks_per_grid;
  dim3 threads_per_block;

  if (length > 1024) {
    blocks_per_grid = dim3(ceil((length) / 1024.0), 1, 1);
    threads_per_block = dim3(1024, 1, 1);
  } else {
    blocks_per_grid = dim3(1, 1, 1);
    threads_per_block = dim3(length, 1, 1);
  }

  cuda_ListArray_num<int32_t, int64_t><<<blocks_per_grid, threads_per_block>>>(
    tonum,
    fromstarts,
    startsoffset,
    fromstops,
    stopsoffset);

  return success();
}
Error
awkward_cuda_ListArrayU32_num_64(
  int64_t* tonum,
  const uint32_t* fromstarts,
  int64_t startsoffset,
  const uint32_t* fromstops,
  int64_t stopsoffset,
  int64_t length) {

  dim3 blocks_per_grid;
  dim3 threads_per_block;

  if (length > 1024) {
    blocks_per_grid = dim3(ceil((length) / 1024.0), 1, 1);
    threads_per_block = dim3(1024, 1, 1);
  } else {
    blocks_per_grid = dim3(1, 1, 1);
    threads_per_block = dim3(length, 1, 1);
  }

  cuda_ListArray_num<uint32_t, int64_t><<<blocks_per_grid, threads_per_block>>>(
    tonum,
    fromstarts,
    startsoffset,
    fromstops,
    stopsoffset);

  return success();
}
Error
awkward_cuda_ListArray64_num_64(
  int64_t* tonum,
  const int64_t* fromstarts,
  int64_t startsoffset,
  const int64_t* fromstops,
  int64_t stopsoffset,
  int64_t length) {

  dim3 blocks_per_grid;
  dim3 threads_per_block;

  if (length > 1024) {
    blocks_per_grid = dim3(ceil((length) / 1024.0), 1, 1);
    threads_per_block = dim3(1024, 1, 1);
  } else {
    blocks_per_grid = dim3(1, 1, 1);
    threads_per_block = dim3(length, 1, 1);
  }

  cuda_ListArray_num<int64_t , int64_t><<<blocks_per_grid, threads_per_block>>>(
    tonum,
    fromstarts,
    startsoffset,
    fromstops,
    stopsoffset);

  return success();
}
