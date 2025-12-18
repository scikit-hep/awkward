// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include <hip/hip_runtime.h>
#include "awkward/kernels.h"

// The actual GPU logic
template <typename OUT, typename IN>
__global__ void hip_reduce_argmax_kernel(
    OUT* toptr,
    const IN* fromptr,
    const int64_t* parents,
    int64_t lenparents) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < lenparents) {
        int64_t parent = parents[i];
        IN value = fromptr[i];

        // This is a "Compare and Swap" loop to find the argmax atomically
        // We need to ensure that only the index of the highest value is stored
        OUT* address = &toptr[parent];
        OUT old_idx, assumed_idx;

        old_idx = *address;
        do {
            assumed_idx = old_idx;
            // If current index is -1 OR current value > value at the stored index
            if (assumed_idx == -1 || value > fromptr[assumed_idx]) {
                old_idx = atomicCAS(address, assumed_idx, (OUT)i);
            } else {
                break; // Current value is not larger, no update needed
            }
        } while (assumed_idx != old_idx);
    }
}

// The initialization kernel
template <typename OUT>
__global__ void hip_reduce_argmax_init(OUT* toptr, int64_t outlength) {
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < outlength) {
        toptr[k] = -1;
    }
}

template <typename OUT, typename IN>
ERROR awkward_reduce_argmax(
    OUT* toptr,
    const IN* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength) {

    // 1. Initialize toptr with -1
    int threads_init = 256;
    int blocks_init = (outlength + threads_init - 1) / threads_init;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_reduce_argmax_init<OUT>), blocks_init, threads_init, 0, 0, 
                       toptr, outlength);

    // 2. Run the argmax reduction
    int threads_red = 256;
    int blocks_red = (lenparents + threads_red - 1) / threads_red;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_reduce_argmax_kernel<OUT, IN>), blocks_red, threads_red, 0, 0, 
                       toptr, fromptr, parents, lenparents);

    return success();
}

