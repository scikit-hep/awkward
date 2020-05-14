#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/cpu-kernels/util.h"

class KernelCore {
public:
    KernelCore() {}

    static ERROR new_identities32(uint8_t memory_loc,
                                  int32_t *toptr,
                                  int64_t length);

    static ERROR new_identities64(uint8_t memory_loc,
                                  int64_t *toptr,
                                  int64_t length);

};


#endif //AWKWARD_KERNEL_H
