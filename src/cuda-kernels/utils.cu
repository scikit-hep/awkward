#include "awkward/cuda-kernels/cuda_utils.h"

ERROR awkward_cuda_ptr_device_num(int& device_num, void* ptr) {
  cudaPointerAttributes att;
  cudaError_t status = cudaPointerGetAttributes(&att, ptr);
  if(status != cudaError::cudaSuccess)
    return failure(cudaGetErrorString(status), 0, kSliceNone);
   device_num = att.device;
  return success();
}

ERROR awkward_cuda_ptr_device_name(std::string& device_name, void* ptr) {
  cudaPointerAttributes att;
  cudaError_t status = cudaPointerGetAttributes(&att, ptr);
  if(status != cudaError::cudaSuccess)
    return failure(cudaGetErrorString(status), 0, kSliceNone);

  cudaDeviceProp dev_prop;
  status = cudaGetDeviceProperties(&dev_prop, att.device);
  if(status != cudaError::cudaSuccess)
    return failure(cudaGetErrorString(status), 0, kSliceNone);
  device_name = dev_prop.name;
  return success();
}

