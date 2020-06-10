// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CUDA_KERNELS_OPERATIONS_CUH
#define AWKWARD_CUDA_KERNELS_OPERATIONS_CUH

#include <stdint.h>
#include "awkward/common.h"

extern "C" {
EXPORT_SYMBOL struct Error
  awkward_cuda_listarray32_num_64(
    int64_t* tonum,
    const int32_t* fromstarts,
    int64_t startsoffset,
    const int32_t* fromstops,
    int64_t stopsoffset,
    int64_t length);
};
#endif //AWKWARD_CUDA_KERNELS_OPERATIONS_CUH
