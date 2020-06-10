// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/allocators.cuh"

extern "C" {

  int awkward_cuda_ptr_loc(void* ptr) {
    cudaPointerAttributes att;
    auto err = cudaPointerGetAttributes(&att, ptr);
    return att.device;
  }

  int8_t *awkward_cuda_ptri8_alloc(int64_t length) {
    if(length != 0) {
      int8_t *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(int8_t) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptri8_dealloc(int8_t* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  uint8_t *awkward_cuda_ptriU8_alloc(int64_t length) {
    if(length != 0) {
      uint8_t *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(uint8_t) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptriU8_dealloc(uint8_t* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  int32_t *awkward_cuda_ptri32_alloc(int64_t length) {
    if(length != 0) {
      int32_t *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(int32_t) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptri32_dealloc(int32_t* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  uint32_t *awkward_cuda_ptriU32_alloc(int64_t length) {
    if(length != 0) {
      uint32_t *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(uint32_t) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptriU32_dealloc(uint32_t* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  int64_t *awkward_cuda_ptri64_alloc(int64_t length) {
    if(length != 0) {
      int64_t *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(int64_t) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptr64_dealloc(int64_t* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  float *awkward_cuda_ptrf_alloc(int64_t length) {
    if(length != 0) {
      float *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(float) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptrf_dealloc(float* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  double *awkward_cuda_ptrd_alloc(int64_t length) {
    if(length != 0) {
      double *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(double) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptrd_dealloc(double* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }

  bool *awkward_cuda_ptrb_alloc(int64_t length) {
    if(length != 0) {
      bool *ptr;
      cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(bool) * length);
      if(err == cudaError::cudaSuccess)
        return ptr;
    }
    return nullptr;
  }

  void *awkward_cuda_ptrb_dealloc(bool* ptr) {
    cudaError_t  err = cudaFree((void *)ptr);
  }
}


