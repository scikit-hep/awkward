// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/kernels/cuda-utils.h"

ERROR awkward_cuda_ptr_device_num(int& device_num, void* ptr) {
  cudaPointerAttributes att;
  cudaError_t status = cudaPointerGetAttributes(&att, ptr);
  if (status != cudaError::cudaSuccess) {
    return failure(cudaGetErrorString(status), 0, kSliceNone, true);
  }
  device_num = att.device;
  return success();
}

ERROR awkward_cuda_ptr_device_name(std::string& device_name, void* ptr) {
  cudaPointerAttributes att;
  cudaError_t status = cudaPointerGetAttributes(&att, ptr);
  if (status != cudaError::cudaSuccess) {
    return failure(cudaGetErrorString(status), 0, kSliceNone, true);
  }

  cudaDeviceProp dev_prop;
  status = cudaGetDeviceProperties(&dev_prop, att.device);
  if (status != cudaError::cudaSuccess) {
    return failure(cudaGetErrorString(status), 0, kSliceNone, true);
  }
  device_name = dev_prop.name;
  return success();
}

ERROR awkward_cuda_host_to_device(
  void* to_ptr,
  void* from_ptr,
  int64_t bytelength) {
  cudaError_t memcpy_stat = cudaMemcpy(to_ptr,
                                       from_ptr,
                                       bytelength,
                                       cudaMemcpyHostToDevice);
  if (memcpy_stat != cudaError_t::cudaSuccess) {
    return failure(cudaGetErrorString(memcpy_stat), kSliceNone, kSliceNone, true);
  }
  else {
    return success();
  }
}

ERROR awkward_cuda_device_to_host(
  void* to_ptr,
  void* from_ptr,
  int64_t bytelength) {
  cudaError_t memcpy_stat = cudMemcpy(to_ptr,
                                      from_ptr,
                                      bytelength,
                                      cudaMemcpyDeviceToHost);
  if (memcpy_stat != cudaError_t::cudaSuccess) {
    return failure(cudaGetErrorString(memcpy_stat), kSliceNone, kSliceNone, true);
  }
  else {
    return success();
  }
}
