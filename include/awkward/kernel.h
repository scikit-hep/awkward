// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_H
#define AWKWARD_KERNEL_H

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/cpu-kernels/allocators.h"
#include "awkward/cpu-kernels/getitem.h"

#ifndef _MSC_VER
  #include "dlfcn.h"
#endif

namespace kernel {
  template<typename T>
  std::shared_ptr<T>
    ptr_alloc(int64_t length,
              KernelsLib ptr_lib);

  template <typename T>
  T
    index_getitem_at_nowrap(T* ptr,
                            int64_t offset,
                            int64_t at,
                            KernelsLib ptr_lib);

  template <typename T>
  void
    index_setitem_at_nowrap(T* ptr,
                            int64_t offset,
                            int64_t at,
                            T value,
                            KernelsLib ptr_lib);

};

#endif //AWKWARD_KERNEL_H
