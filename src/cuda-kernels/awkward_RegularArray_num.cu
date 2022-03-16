// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/manual_awkward_ListArray_num.cu", line)

// #include "awkward/kernels.h"
#include <cstdint>
template <typename T>
__global__ void
cuda_RegularArray_num(T* tonum, int64_t size, int64_t length, int64_t invocation_index, int64_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id == bitmasklength && !(err_code[0] % 8)) {
    err_code[0] = invocation_index + 8;
  }
  if (thread_id < length && !(err_code[0] % 8)) {
    tonum[thread_id] = size;
  }
}
