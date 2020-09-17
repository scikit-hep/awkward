#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_ListOffsetArray_rpad_and_clip_axis1.cu", line)

#include "awkward/kernels/operations.h"
#include "standard_parallel_algorithms.h"
//
//template <typename T, typename C>
//__device__ void
//nested_loop_j_1(
//    T* toindex,
//    const C* fromoffsets,
//    int64_t length,
//    int64_t target,
//    int64_t i) {
//
//  int64_t block_id =
//      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//  int64_t thread_id = block_id * blockDim.x + threadIdx.x;
//
//  if(thread_id < length) {
//    toindex[i * target + thread_id] = (T)fromoffsets[i] + thread_id;
//  }
//}
//
//
//template <typename T, typename C>
//__device__ void
//nested_loop_j_2(
//    T* toindex,
//    int64_t shorter,
//    int64_t target,
//    int64_t i) {
//
//  int64_t block_id =
//      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//  int64_t thread_id = block_id * blockDim.x + threadIdx.x;
//
//  if(thread_id <= (target - shorter)) {
//    toindex[i*target + (thread_id + shorter)] = -1;
//  }
//}
//
//

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_rpad_and_clip_axis1_kernel(
    T* toindex,
    const C* fromoffsets,
    int64_t length,
    int64_t target) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int64_t thread_id = block_id * blockDim.x + threadIdx.x;

  if(thread_id < length) {
    int64_t rangeval = (T)(fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
    int64_t shorter = (target < rangeval) ? target : rangeval;

    for (int64_t j = 0; j < shorter; j++) {
      toindex[thread_id * target + j] = (T)fromoffsets[thread_id] + j;
    }
    for (int64_t j = shorter; j < target; j++) {
      toindex[thread_id * target + j] = -1;
    }

  }
}

template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_and_clip_axis1(
    T* toindex,
    const C* fromoffsets,
    int64_t length,
    int64_t target) {

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  awkward_ListOffsetArray_rpad_and_clip_axis1_kernel<T, C><<<blocks_per_grid, threads_per_block>>>(
      toindex,
      fromoffsets,
      length,
      target);

  cudaDeviceSynchronize();

  return success();
}
ERROR awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const int32_t* fromoffsets,
    int64_t length,
    int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int32_t>(
      toindex,
      fromoffsets,
      length,
      target);
}
ERROR awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const uint32_t* fromoffsets,
    int64_t length,
    int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, uint32_t>(
      toindex,
      fromoffsets,
      length,
      target);
}
ERROR awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const int64_t* fromoffsets,
    int64_t length,
    int64_t target) {
  return awkward_ListOffsetArray_rpad_and_clip_axis1<int64_t, int64_t>(
      toindex,
      fromoffsets,
      length,
      target);
}
