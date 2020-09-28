// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CUDA_UTILS_H
#define AWKWARD_CUDA_UTILS_H

#include "awkward/common.h"

extern "C" {
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
