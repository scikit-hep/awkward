#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/manual_awkward_ListOffsetArray_rpad_axis1.cu", line)

#include "awkward/kernels/operations.h"
#include "standard_parallel_algorithms.h"

template <typename T, typename C>
__global__
void awkward_ListOffsetArray_rpad_axis1_kernel(
  T* toindex,
  const C* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t thread_idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(thread_idx < fromlength) {
    int64_t rangeval = (T)(fromoffsets[thread_idx + 1] - fromoffsets[thread_idx]);

	  if(thread_idy < rangeval) {
      toindex[thread_idx * target + thread_idy] = (T)fromoffsets[thread_idx] + thread_idy;
    }
	  else if(thread_idy >= rangeval && thread_idy < target) {
      toindex[thread_idx * target + thread_idy] = -1;
    }
  }
}
    
template <typename T, typename C>
ERROR awkward_ListOffsetArray_rpad_axis1(
  T* toindex,
  const C* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  dim3 blocks_per_grid = blocks_2d(fromlength, target);
  dim3 threads_per_block = threads_2d(fromlength, target);

  awkward_ListOffsetArray_rpad_axis1_kernel<T, C><<<blocks_per_grid, threads_per_block>>>(
      toindex,
      fromoffsets,
      fromlength,
      target);

  cudaDeviceSynchronize();
  return success();
}
ERROR awkward_ListOffsetArray32_rpad_axis1_64(
  int64_t* toindex,
  const int32_t* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, int32_t>(
    toindex,
    fromoffsets,
    fromlength,
    target);
}
ERROR awkward_ListOffsetArrayU32_rpad_axis1_64(
  int64_t* toindex,
  const uint32_t* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, uint32_t>(
    toindex,
    fromoffsets,
    fromlength,
    target);
}
ERROR awkward_ListOffsetArray64_rpad_axis1_64(
  int64_t* toindex,
  const int64_t* fromoffsets,
  int64_t fromlength,
  int64_t target) {
  return awkward_ListOffsetArray_rpad_axis1<int64_t, int64_t>(
    toindex,
    fromoffsets,
    fromlength,
    target);
}
