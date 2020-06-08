// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"
#include "awkward/cpu-kernels/allocators.h"

namespace kernel {
    template<typename T>
    T* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <>
    int8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <>
    uint8_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <>
    int32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <>
    uint32_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <>
    int64_t* ptr_alloc(int64_t length, KernelsLib ptr_lib);

    template <>
    double* ptr_alloc(int64_t length, KernelsLib ptr_lib);



    template <typename T>
    ERROR new_identities(T *toptr,
                         int64_t length);
};

#endif //AWKWARD_KERNEL_H
