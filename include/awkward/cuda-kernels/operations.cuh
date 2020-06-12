// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CUDA_KERNELS_OPERATIONS_CUH
#define AWKWARD_CUDA_KERNELS_OPERATIONS_CUH

#include <stdint.h>
#include "awkward/common.h"

extern "C" {
  void
  awkward_cuda_listarray8_num_32(
    int32_t* tonum,
    const int8_t* fromstarts,
    int32_t startsoffset,
    const int8_t* fromstops,
    int32_t stopsoffset,
    int32_t length);
};
#endif //AWKWARD_CUDA_KERNELS_OPERATIONS_CUH
