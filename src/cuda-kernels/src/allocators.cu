// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/cuda_allocators.h"

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

template <typename  T>
Error awkward_cuda_ptr_dealloc(const T* ptr) {
  cudaError_t  status = cudaFree((void *)ptr);
  if(status != cudaError_t::cudaSuccess) {
      std::cout << "Reached3" << "\n";
      return failure(cudaGetErrorString(status), 0, kSliceNone);
  }

  return success();
}
ERROR awkward_cuda_ptrbool_dealloc(const bool *ptr) {
  return awkward_cuda_ptr_dealloc<bool>(ptr);
}
ERROR awkward_cuda_ptr8_dealloc(const int8_t *ptr) {
  return awkward_cuda_ptr_dealloc<int8_t>(ptr);
}
ERROR awkward_cuda_ptrU8_dealloc(const uint8_t *ptr) {
  return awkward_cuda_ptr_dealloc<uint8_t>(ptr);
}
ERROR awkward_cuda_ptr16_dealloc(const int16_t *ptr) {
  return awkward_cuda_ptr_dealloc<int16_t>(ptr);
}
ERROR awkward_cuda_ptrU16_dealloc(const uint16_t *ptr) {
  return awkward_cuda_ptr_dealloc<uint16_t>(ptr);
}
ERROR awkward_cuda_ptr32_dealloc(const int32_t *ptr) {
  return awkward_cuda_ptr_dealloc<int32_t>(ptr);
}
ERROR awkward_cuda_ptrU32_dealloc(const uint32_t *ptr) {
  return awkward_cuda_ptr_dealloc<uint32_t>(ptr);
}
ERROR awkward_cuda_ptr64_dealloc(const int64_t *ptr) {
  return awkward_cuda_ptr_dealloc<int64_t>(ptr);
}
ERROR awkward_cuda_ptrU64_dealloc(const uint64_t *ptr) {
  return awkward_cuda_ptr_dealloc<uint64_t>(ptr);
}
ERROR awkward_cuda_ptrfloat32_dealloc(const float *ptr) {
  return awkward_cuda_ptr_dealloc<float>(ptr);
}
ERROR awkward_cuda_ptrfloat64_dealloc(const double *ptr) {
  return awkward_cuda_ptr_dealloc<double>(ptr);
}

template <typename T>
ERROR awkward_cuda_H2D(
  T** to_ptr,
  T* from_ptr,
  int64_t length) {
  cudaError_t malloc_stat = cudaMallocManaged((void**)to_ptr,
                                          sizeof(T) * length);
  if(malloc_stat != cudaError_t::cudaSuccess) {
      return failure(cudaGetErrorString(malloc_stat), 0,kSliceNone);
  }
  cudaError_t memcpy_stat = cudaMemcpy(*to_ptr,
                                       from_ptr,
                                       sizeof(T) * length,
                                       cudaMemcpyHostToDevice);

  if(memcpy_stat != cudaError_t::cudaSuccess) {
      return failure(cudaGetErrorString(memcpy_stat), 0, kSliceNone);
  }
  return success();
}
ERROR awkward_cuda_H2Dbool(
  bool **to_ptr,
  bool *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<bool>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2D8(
  int8_t **to_ptr,
  int8_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<int8_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2DU8(
  uint8_t **to_ptr,
  uint8_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<uint8_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2D16(
  int16_t **to_ptr,
  int16_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<int16_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2DU16(
  uint16_t **to_ptr,
  uint16_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<uint16_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2D32(
  int32_t **to_ptr,
  int32_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<int32_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2DU32(
  uint32_t **to_ptr,
  uint32_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<uint32_t>(
    to_ptr,
    from_ptr,
    length);
}
Error awkward_cuda_H2D64(
  int64_t **to_ptr,
  int64_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<int64_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2DU64(
  uint64_t **to_ptr,
  uint64_t *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<uint64_t>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2Dfloat32(
  float **to_ptr,
  float *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<float>(
    to_ptr,
    from_ptr,
    length);
}
ERROR awkward_cuda_H2Dfloat64(
  double **to_ptr,
  double *from_ptr,
  int64_t length) {
  return awkward_cuda_H2D<double>(
    to_ptr,
    from_ptr,
    length);
}

template <typename T>
ERROR awkward_cuda_D2H(
        T* to_ptr,
        T* from_ptr,
        int64_t length) {
    cudaError_t memcpy_stat = cudaMemcpy(to_ptr,
                                         from_ptr,
                                         sizeof(T) * length,
                                         cudaMemcpyDeviceToHost);

    if(memcpy_stat != cudaError_t::cudaSuccess) {
        return failure(cudaGetErrorString(memcpy_stat), 0, kSliceNone);
    }
    return success();
}
ERROR awkward_cuda_D2Hbool(
        bool *to_ptr,
        bool *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<bool>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2H8(
        int8_t *to_ptr,
        int8_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<int8_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2HU8(
        uint8_t *to_ptr,
        uint8_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<uint8_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2H16(
        int16_t *to_ptr,
        int16_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<int16_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2HU16(
        uint16_t *to_ptr,
        uint16_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<uint16_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2H32(
        int32_t *to_ptr,
        int32_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<int32_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2HU32(
        uint32_t *to_ptr,
        uint32_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<uint32_t>(
            to_ptr,
            from_ptr,
            length);
}
Error awkward_cuda_D2H64(
        int64_t *to_ptr,
        int64_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<int64_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2HU64(
        uint64_t *to_ptr,
        uint64_t *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<uint64_t>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2Hfloat32(
        float *to_ptr,
        float *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<float>(
            to_ptr,
            from_ptr,
            length);
}
ERROR awkward_cuda_D2Hfloat64(
        double *to_ptr,
        double *from_ptr,
        int64_t length) {
    return awkward_cuda_D2H<double>(
            to_ptr,
            from_ptr,
            length);
}

