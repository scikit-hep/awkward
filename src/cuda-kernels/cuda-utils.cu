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

