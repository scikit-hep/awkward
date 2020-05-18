//
// Created by root on 5/14/20.
//

#ifndef AWKWARDGPU_IDENTITIES_H
#define AWKWARDGPU_IDENTITIES_H

#include "awkward/cpu-kernels/util.h"

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
