// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"
#include "awkward/Index.h"

namespace kernel {
    template<typename T>
    T* get_ptr(T* array, KernelsLib input_array_1, KernelsLib input_array_2) {
      if(input_array_1 == cpu_kernels && input_array_2 == cpu_kernels) {
        return array;
      }
      else {
        // dynamically load cuda_kernels to get appropriate pointer
      }
    }

    template <typename T>
        ERROR new_identities(T *toptr,
                         int64_t length);
};

#endif //AWKWARD_KERNEL_H
