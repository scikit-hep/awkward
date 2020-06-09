// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ALLOCATORS_H_
#define AWKWARD_ALLOCATORS_H_

#include <cstdint>
#include <stdlib.h>

extern "C" {
  int8_t *awkward_cpu_ptri8_alloc(int64_t length);

  uint8_t *awkward_cpu_ptriU8_alloc(int64_t length);

  int32_t *awkward_cpu_ptri32_alloc(int64_t length);

  uint32_t *awkward_cpu_ptriU32_alloc(int64_t length);

  int64_t *awkward_cpu_ptri64_alloc(int64_t length);

  float *awkward_cpu_ptrf_alloc(int64_t length);

  double *awkward_cpu_ptrd_alloc(int64_t length);

  bool *awkward_cpu_ptrb_alloc(int64_t length);
};
#endif //AWKWARD_ALLOCATORS_H_
