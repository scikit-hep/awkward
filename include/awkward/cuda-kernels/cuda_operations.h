// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CUDA_KERNELS_OPERATIONS_H
#define AWKWARD_CUDA_KERNELS_OPERATIONS_H

#include <stdint.h>
#include "awkward/common.h"

extern "C" {
  EXPORT_SYMBOL Error
  awkward_cuda_ListArray32_num_64(
    int64_t* tonum,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL Error
  awkward_cuda_ListArrayU32_num_64(
    int64_t* tonum,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL Error
  awkward_cuda_ListArray64_num_64(
    int64_t* tonum,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL Error
  awkward_cuda_RegularArray_num_64(
    int64_t* tonum,
    int64_t size,
    int64_t length);
}

#endif //AWKWARD_CUDA_KERNELS_OPERATIONS_H
