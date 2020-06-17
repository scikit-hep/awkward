// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ALLOCATORS_H_
#define AWKWARD_ALLOCATORS_H_

#include <cstdint>
#include <stdlib.h>

extern "C" {
  int8_t *awkward_cpu_ptr8_alloc(int64_t length);
  uint8_t *awkward_cpu_ptrU8_alloc(int64_t length);
  int16_t *awkward_cpu_ptr16_alloc(int64_t length);
  uint16_t *awkward_cpu_ptrU16_alloc(int64_t length);
  int32_t *awkward_cpu_ptr32_alloc(int64_t length);
  uint32_t *awkward_cpu_ptrU32_alloc(int64_t length);
  int64_t *awkward_cpu_ptr64_alloc(int64_t length);
  uint64_t *awkward_cpu_ptrU64_alloc(int64_t length);
  float *awkward_cpu_ptrfloat32_alloc(int64_t length);
  double *awkward_cpu_ptrfloat64_alloc(int64_t length);
  bool *awkward_cpu_ptrbool_alloc(int64_t length);
};
#endif //AWKWARD_ALLOCATORS_H_
