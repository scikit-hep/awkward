// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"

namespace kernel {
    template <typename T>
    ERROR new_identities(T *toptr,
                         int64_t length);
};

#endif //AWKWARD_KERNEL_H
