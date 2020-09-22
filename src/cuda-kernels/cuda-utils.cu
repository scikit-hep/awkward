// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/cuda-utils.cu", line)

#include "awkward/cuda-utils.h"

ERROR awkward_cuda_ptr_device_num(int64_t* num, void* ptr) {
  cudaPointerAttributes att;
  cudaError_t status = cudaPointerGetAttributes(&att, ptr);
  if (status != cudaError::cudaSuccess) {
    return failure_pass_through(cudaGetErrorString(status), kSliceNone, kSliceNone, FILENAME(__LINE__));
  }
  *num = att.device;
  return success();
}

ERROR awkward_cuda_ptr_device_name(char* name, void* ptr) {
  cudaPointerAttributes att;
  cudaError_t status = cudaPointerGetAttributes(&att, ptr);
  if (status != cudaError::cudaSuccess) {
    return failure_pass_through(cudaGetErrorString(status), kSliceNone, kSliceNone, FILENAME(__LINE__));
  }
  cudaDeviceProp dev_prop;
  status = cudaGetDeviceProperties(&dev_prop, att.device);
  if (status != cudaError::cudaSuccess) {
    return failure_pass_through(cudaGetErrorString(status), kSliceNone, kSliceNone, FILENAME(__LINE__));
  }
  strcpy(name, dev_prop.name);
  return success();
}

ERROR awkward_cuda_host_to_device(
  void* to_ptr,
  void* from_ptr,
  int64_t bytelength) {
  cudaError_t memcpy_stat = cudaMemcpy(
    to_ptr, from_ptr, bytelength, cudaMemcpyHostToDevice);
  if (memcpy_stat != cudaError_t::cudaSuccess) {
    return failure_pass_through(cudaGetErrorString(memcpy_stat), kSliceNone, kSliceNone, FILENAME(__LINE__));
  }
  else {
    return success();
  }
}

ERROR awkward_cuda_device_to_host(
  void* to_ptr,
  void* from_ptr,
  int64_t bytelength) {
  cudaError_t memcpy_stat = cudaMemcpy(to_ptr,
                                       from_ptr,
                                       bytelength,
                                       cudaMemcpyDeviceToHost);
  if (memcpy_stat != cudaError_t::cudaSuccess) {
    return failure_pass_through(cudaGetErrorString(memcpy_stat), kSliceNone, kSliceNone, FILENAME(__LINE__));
  }
  else {
    return success();
  }
}
