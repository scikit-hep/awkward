// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/allocators.cpp", line)

#include "awkward/kernel-utils.h"

void* awkward_malloc(int64_t bytelength) {
  if (bytelength == 0) {
    return nullptr;
  }
  else {
    uint8_t* out = new uint8_t[bytelength];
    return reinterpret_cast<void*>(out);
  }
}

void awkward_free(void const *ptr) {
  uint8_t const* in = reinterpret_cast<uint8_t const*>(ptr);
  delete [] in;
}
