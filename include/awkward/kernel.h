// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"
#include "awkward/Index.h"

namespace kernel {
    template<typename T, typename F, typename U>
    T* get_ptr(T* array, F input_array_1, U input_array_2) {
      if(input_array_1.ptr_lib == cpu_kernels && input_array_2.ptr_lib == cpu_kernels) {
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
