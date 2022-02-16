// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_CUDA_FUNCTIONS_H
#define AWKWARD_CUDA_FUNCTIONS_H

#include "awkward/common.h"

extern "C" {
  inline dim3
  threads(int64_t length) {
    if (length > 1024) {
      return dim3(1024);
    }
    return dim3(length);
  }
  
  inline dim3
  blocks(int64_t length) {
    if (length > 1024) {
      return dim3(ceil((length) / 1024.0));
    }
    return dim3(1);
  }
  
  inline dim3
  threads_2d(int64_t length_x, int64_t length_y) {
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
  
  inline dim3
  blocks_2d(int64_t length_x, int64_t length_y) {
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

  inline ERROR post_kernel_checks(ERROR* kernel_err = nullptr) {
    ERROR err;
    if(kernel_err != nullptr) {
        err = *kernel_err;
    }
    else {
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
          err = failure(
              cudaGetErrorString(cuda_err), kSliceNone, kSliceNone, FILENAME(__LINE__));
        }
        err = success();
    }
    cudaDeviceSynchronize();
    return err;
  }
}

#endif //AWKWARD_CUDA_UTILS_H
