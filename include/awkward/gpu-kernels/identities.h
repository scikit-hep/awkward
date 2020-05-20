#ifndef AWKWARDGPU_IDENTITIES_H
#define AWKWARDGPU_IDENTITIES_H

#include "awkward/common_utils.h"

extern "C" {
EXPORT_SYMBOL struct Error
awkward_gpu_new_identities32(uint8_t memory_loc,
                             int32_t *toptr,
                             int64_t length);
EXPORT_SYMBOL struct Error
awkward_gpu_new_identities64(uint8_t memory_loc,
                             int64_t *toptr,
                             int64_t length);
}
#endif //AWKWARDGPU_IDENTITIES_H
