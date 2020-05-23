#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common_utils.h"

namespace kernel
{

    template <typename T>
    ERROR new_identities(T *toptr,
                         int64_t length);

};

#endif //AWKWARD_KERNEL_H
