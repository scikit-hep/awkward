// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_CUDA_UTILS_H
#define AWKWARD_CUDA_UTILS_H

#include "awkward/common.h"

extern "C" {
  dim3 threads(int64_t length);
  
  dim3 blocks(int64_t length);
  
  dim3 threads_2d(int64_t length_x, int64_t length_y);
  
  dim3 blocks_2d(int64_t length_x, int64_t length_y);
  
  ERROR post_kernel_checks(ERROR* kernel_err = nullptr);

  EXPORT_SYMBOL ERROR awkward_cuda_ptr_device_num(
    int64_t* device_num,
    void* ptr);

  EXPORT_SYMBOL ERROR awkward_cuda_ptr_device_name(
    char* name,
    void* ptr);

  EXPORT_SYMBOL ERROR awkward_cuda_host_to_device(
    void* to_ptr,
    void* from_ptr,
    int64_t bytelength);

  EXPORT_SYMBOL ERROR awkward_cuda_device_to_host(
    void* to_ptr,
    void* from_ptr,
    int64_t bytelength);

}

#endif //AWKWARD_CUDA_UTILS_H
