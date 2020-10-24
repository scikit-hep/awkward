#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_index_rpad_and_clip_axis1.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

template <typename T>
__global__ void
awkward_index_rpad_and_clip_axis1_kernel(
    T* tostarts,
    T* tostops,
    int64_t target,
    int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    tostarts[thread_id] = thread_id * target;
    tostops[thread_id] = (thread_id + 1) * target;
  }
}

template <typename T>
ERROR awkward_index_rpad_and_clip_axis1(
    T* tostarts,
    T* tostops,
    int64_t target,
    int64_t length) {

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  awkward_index_rpad_and_clip_axis1_kernel<<<blocks_per_grid, threads_per_block>>>(
      tostarts,
      tostops,
      target,
      length);

  cudaDeviceSynchronize();

  return success();
}
ERROR awkward_index_rpad_and_clip_axis1_64(
    int64_t* tostarts,
    int64_t* tostops,
    int64_t target,
    int64_t length) {
  return awkward_index_rpad_and_clip_axis1<int64_t>(
      tostarts,
      tostops,
      target,
      length);
}
