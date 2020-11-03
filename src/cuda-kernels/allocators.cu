// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/allocators.cu", line)

#include "awkward/kernel-utils.h"

void* awkward_malloc(int64_t bytelength) {
  void* out = nullptr;
  if (bytelength != 0) {
    cudaError_t err = cudaMallocManaged(&out, bytelength);
    if (err != cudaError::cudaSuccess) {
      out = nullptr;
    }
  }
  return out;
}

void awkward_free(void const *ptr) {
  cudaFree((void*)ptr);
}
