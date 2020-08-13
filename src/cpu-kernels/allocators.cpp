// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/kernels/allocators.h"

void* awkward_malloc(int64_t length) {
  if (length == 0) {
    return nullptr;
  }
  else {
    uint8_t* out = new uint8_t[length];
    return reinterpret_cast<void*>(out);
  }
}

ERROR awkward_free(const void* ptr) {
  const uint8_t* in = reinterpret_cast<const uint8_t*>(ptr);
  delete [] in;
  return success();
}
