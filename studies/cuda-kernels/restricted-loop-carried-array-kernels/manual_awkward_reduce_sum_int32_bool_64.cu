// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_reduce_sum_int32_bool_64.cu", line)

#include "standard_parallel_algorithms.h"
#include "awkward/kernels.h"

__global__ void
awkward_reduce_sum_int32_bool_64_kernel(int32_t* toptr,
                                        const bool* fromptr,
                                        const int64_t* parents,
                                        int64_t lenparents) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < lenparents) {
    toptr[parents[thread_id]] += (fromptr[thread_id] != 0);
  }
}

ERROR
awkward_reduce_sum_int32_bool_64(int32_t* toptr,
                                 const bool* fromptr,
                                 const int64_t* parents,
                                 int64_t lenparents,
                                 int64_t outlength) {
  HANDLE_ERROR(cudaMemset(toptr, 0, sizeof(int32_t) * outlength));

  dim3 blocks_per_grid = blocks(lenparents);
  dim3 threads_per_block = threads(lenparents);

  awkward_reduce_sum_int32_bool_64_kernel<<<blocks_per_grid, threads_per_block>>>(
      toptr, fromptr, parents, lenparents);

  return success();
}
