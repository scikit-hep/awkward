// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/allocators.h"

extern "C" {
  int8_t *awkward_cpu_ptri8_alloc(int64_t length) {
    if(length != 0) {
      return (int8_t *)(malloc(sizeof(int8_t) * length));
    }
    return nullptr;
  }

  uint8_t *awkward_cpu_ptriU8_alloc(int64_t length) {
    if(length != 0) {
      return (uint8_t *)(malloc(sizeof(uint8_t) * length));
    }
    return nullptr;
  }

  int32_t *awkward_cpu_ptri32_alloc(int64_t length) {
    if(length != 0) {
      return (int32_t *)(malloc(sizeof(int32_t) * length));
    }
    return nullptr;
  }

  uint32_t *awkward_cpu_ptriU32_alloc(int64_t length) {
    if(length != 0) {
      return (uint32_t *)(malloc(sizeof(uint32_t) * length));
    }
    return nullptr;
  }

  int64_t *awkward_cpu_ptri64_alloc(int64_t length) {
    if(length != 0) {
      return (int64_t *)(malloc(sizeof(int64_t) * length));
    }
    return nullptr;
  }

  double *awkward_cpu_ptrd_alloc(int64_t length) {
    if(length != 0) {
      return (double *)(malloc(sizeof(double) * length));
    }
    return nullptr;
  }
}