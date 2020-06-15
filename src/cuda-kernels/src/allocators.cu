// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/allocators.cuh"

int awkward_cuda_ptr_loc(void* ptr) {
  cudaPointerAttributes att;
  auto err = cudaPointerGetAttributes(&att, ptr);
  return att.device;
}

int8_t* awkward_cuda_host_to_device_buffi8_transfer (int8_t* ptr, int64_t length) {
  int8_t* cuda_ptr;
  cudaError_t err = cudaMallocManaged((void**)&cuda_ptr, sizeof(int8_t) * length);
  cudaError_t err_1 = cudaMemcpy(cuda_ptr, ptr, sizeof(int8_t) * length, cudaMemcpyHostToDevice);
  return cuda_ptr;
}

template <typename  T>
T *awkward_cuda_ptr_alloc(int64_t length) {
  if(length != 0) {
    T *ptr;
    cudaError_t err = cudaMallocManaged((void **)&ptr, sizeof(T) * length);
    if(err == cudaError::cudaSuccess)
      return ptr;
  }
  return nullptr;
}
bool *awkward_cuda_ptrbool_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<bool>(length);
}
int8_t *awkward_cuda_ptr8_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<int8_t>(length);
}
uint8_t *awkward_cuda_ptrU8_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<uint8_t>(length);
}
int16_t *awkward_cuda_ptr16_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<int16_t>(length);
}
uint16_t *awkward_cuda_ptrU16_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<uint16_t>(length);
}
int32_t *awkward_cuda_ptr32_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<int32_t>(length);
}
uint32_t *awkward_cuda_ptrU32_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<uint32_t>(length);
}
int64_t *awkward_cuda_ptr64_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<int64_t>(length);
}
uint64_t *awkward_cuda_ptrU64_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<uint64_t>(length);
}
float *awkward_cuda_ptrfloat32_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<float>(length);
}
double *awkward_cuda_ptrfloat64_alloc(int64_t length) {
  return awkward_cuda_ptr_alloc<double>(length);

}

void *awkward_cuda_ptri8_dealloc(int8_t* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



void *awkward_cuda_ptriU8_dealloc(uint8_t* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



void *awkward_cuda_ptri32_dealloc(int32_t* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



void *awkward_cuda_ptriU32_dealloc(uint32_t* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



void *awkward_cuda_ptr64_dealloc(int64_t* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



void *awkward_cuda_ptrf_dealloc(float* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}


void *awkward_cuda_ptrd_dealloc(double* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



void *awkward_cuda_ptrb_dealloc(bool* ptr) {
  cudaError_t  err = cudaFree((void *)ptr);
}



