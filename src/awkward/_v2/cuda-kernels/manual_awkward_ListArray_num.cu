// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/manual_awkward_ListArray_num.cu", line)

// #include "awkward/kernels.h"
#include <cstdint>
template <typename T, typename C>
__global__ void
cuda_ListArray_num(C* tonum, const T* fromstarts, const T* fromstops, int64_t length) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < length) {
    int64_t start = fromstarts[thread_id];
    int64_t stop = fromstops[thread_id];
    tonum[thread_id] = (C)(stop - start);
  }
}
