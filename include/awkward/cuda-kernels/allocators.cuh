//
// Created by trickarcher on 09/06/20.
//

#ifndef AWKWARD_ALLOCATORS_CUH_
#define AWKWARD_ALLOCATORS_CUH_

#include <stdint.h>

extern "C" {
  int awkward_cuda_ptr_loc(int8_t* ptr);

  int8_t *awkward_cuda_ptri8_alloc(int64_t length);

  void *awkward_cuda_ptri8_dealloc(int8_t* ptr);
  
  uint8_t *awkward_cuda_ptriU8_alloc(int64_t length);

  void *awkward_cuda_ptriU8_dealloc(uint8_t* ptr);
  
  int32_t *awkward_cuda_ptri32_alloc(int64_t length);

  void *awkward_cuda_ptri32_dealloc(int32_t* ptr);

  
  uint32_t *awkward_cuda_ptriU32_alloc(int64_t length);

  void *awkward_cuda_ptrU32_dealloc(uint32_t* ptr);
  
  int64_t *awkward_cuda_ptri64_alloc(int64_t length);

  void *awkward_cuda_ptri64_dealloc(int64_t* ptr);
  
  float *awkward_cuda_ptrf_alloc(int64_t length);

  void *awkward_cuda_ptrf_dealloc(float* ptr);
  
  double *awkward_cuda_ptrd_alloc(int64_t length);

  void *awkward_cuda_ptrd_dealloc(double* ptr);
  
  bool *awkward_cuda_ptrb_alloc(int64_t length);

  void *awkward_cuda_ptrb_dealloc(bool* ptr);
};

#endif //AWKWARD_ALLOCATORS_CUH_
