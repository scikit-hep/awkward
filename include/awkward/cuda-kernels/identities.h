#ifndef AWKWARD_CUDA_IDENTITIES_H
#define AWKWARD_CUDA_IDENTITIES_H

#include "awkward/common_utils.h"

extern "C"
{
    EXPORT_SYMBOL struct Error
    awkward_cuda_new_identities32(int32_t *toptr,
                                  int64_t length);
    EXPORT_SYMBOL struct Error
    awkward_cuda_new_identities64(int64_t *toptr,
                                  int64_t length);
}
#endif //AWKWARD_CUDA_IDENTITIES_H
