#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/cpu-kernels/util.h"

namespace kernel {

template<typename T>
ERROR new_identities(int64_t memory_loc,
                     T *toptr,
                     int64_t length);

};

#endif //AWKWARD_KERNEL_H
