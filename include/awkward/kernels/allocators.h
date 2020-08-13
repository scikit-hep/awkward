// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNELS_ALLOCATORS_H_
#define AWKWARD_KERNELS_ALLOCATORS_H_

#include <cstdint>
#include <stdlib.h>
#include "awkward/common.h"

extern "C" {
  EXPORT_SYMBOL void* awkward_malloc(int64_t bytelength);
  EXPORT_SYMBOL void awkward_free(void const *ptr);
};

#endif //AWKWARD_KERNELS_ALLOCATORS_H_
