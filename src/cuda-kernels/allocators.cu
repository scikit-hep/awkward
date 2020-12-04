// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/allocators.cu", line)

#include "awkward/kernel-utils.h"

void* awkward_malloc(int64_t bytelength) {
  if (bytelength == 0) {
    // std::cout << "CUDA malloc at nullptr (0 bytes)" << std::endl;
    return nullptr;
  }
  else {
    void* out = nullptr;
    cudaError_t err = cudaMallocManaged(&out, bytelength);
    if (err != cudaError::cudaSuccess) {
      // std::cout << "CUDA malloc failed (" << bytelength << " bytes)" << std::endl;
      return nullptr;
    }
    // std::cout << "CUDA malloc at " << out << " (" << bytelength << " bytes)" << std::endl;
    return out;
  }
}

void awkward_free(void const *ptr) {
  // std::cout << "CUDA free   at " << ptr << std::endl;
  cudaFree((void*)ptr);
}
