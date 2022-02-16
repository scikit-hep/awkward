// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_CUDA("src/cuda-kernels/cuda-utils.cu", line)

#include "awkward-cuda/cuda-utils.h"

dim3 threads(int64_t length) {
  if (length > 1024) {
    return dim3(1024);
  }
  return dim3(length);
}

dim3 blocks(int64_t length) {
  if (length > 1024) {
    return dim3(ceil((length) / 1024.0));
  }
  return dim3(1);
}

dim3 threads_2d(int64_t length_x, int64_t length_y) {
  if (length_x > 32 && length_y > 32) {
    return dim3(32, 32);
  } else if (length_x > 32 && length_y <= 32) {
    return dim3(32, length_y);
  } else if (length_x <= 32 && length_y > 32) {
    return dim3(length_x, 32);
  } else {
    return dim3(length_x, length_y);
  }
}

dim3 blocks_2d(int64_t length_x, int64_t length_y) {
  if (length_x > 32 && length_y > 32) {
    return dim3(ceil(length_x / 32.0), ceil(length_y / 32.0));
  } else if (length_x > 32 && length_y <= 32) {
    return dim3(ceil(length_x / 32.0), 1);
  } else if (length_x <= 32 && length_y > 32) {
    return dim3(1, ceil(length_y / 32.0));
  } else {
    return dim3(1, 1);
  }
}

ERROR post_kernel_checks(ERROR* kernel_err = nullptr) {
    ERROR err;
    if(kernel_err != nullptr) {
        err = *kernel_err;
    }
    else {
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
          err = failure(
              cudaGetErrorString(err), kSliceNone, kSliceNone, FILENAME(__LINE__));
        }
        err = success();
    }
    cudaDeviceSynchronize();
    return err;
}

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
