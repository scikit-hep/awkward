// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ALLOCATORS_CUH_
#define AWKWARD_ALLOCATORS_CUH_

#include <stdint.h>
#include "awkward/common.h"

extern "C" {
  int awkward_cuda_ptr_loc(void* ptr);

  bool *awkward_cuda_ptrbool_alloc(int64_t length);
  int8_t *awkward_cuda_ptr8_alloc(int64_t length);
  uint8_t *awkward_cuda_ptrU8_alloc(int64_t length);
  int16_t *awkward_cuda_ptr16_alloc(int64_t length);
  uint16_t *awkward_cuda_ptrU16_alloc(int64_t length);
  int32_t *awkward_cuda_ptr32_alloc(int64_t length);
  uint32_t *awkward_cuda_ptrU32_alloc(int64_t length);
  int64_t *awkward_cuda_ptr64_alloc(int64_t length);
  uint64_t *awkward_cuda_ptrU64_alloc(int64_t length);
  float *awkward_cuda_ptrfloat32_alloc(int64_t length);
  double *awkward_cuda_ptrfloat64_alloc(int64_t length);

  Error awkward_cuda_ptrbool_dealloc(const bool* ptr);
  Error awkward_cuda_ptr8_dealloc(const int8_t* ptr);
  Error awkward_cuda_ptrU8_dealloc(const uint8_t* ptr);
  Error awkward_cuda_ptr16_dealloc(const int16_t* ptr);
  Error awkward_cuda_ptrU16_dealloc(const uint16_t* ptr);
  Error awkward_cuda_ptr32_dealloc(const int32_t* ptr);
  Error awkward_cuda_ptrU32_dealloc(const uint32_t* ptr);
  Error awkward_cuda_ptr64_dealloc(const int64_t* ptr);
  Error awkward_cuda_ptrU64_dealloc(const uint64_t* ptr);
  Error awkward_cuda_ptrfloat32_dealloc(const float* ptr);
  Error awkward_cuda_ptrfloat64_dealloc(const double* ptr);

  Error awkward_cuda_H2D_bool(
    bool** to_ptr,
    bool* from_ptr,
    int64_t length);
  Error awkward_cuda_H2D_8(
    int8_t** to_ptr,
    int8_t* from_ptr,
    int64_t length);
  Error awkward_cuda_H2D_U8(
      uint8_t** to_ptr,
      uint8_t* from_ptr,
      int64_t length);
  Error awkward_cuda_H2D_16(
      int16_t** to_ptr,
      int16_t* from_ptr,
      int64_t length);
  Error awkward_cuda_H2D_U16(
      uint16_t** to_ptr,
      uint16_t* from_ptr,
      int64_t length);
  Error awkward_cuda_H2D_32(
      int32_t** to_ptr,
      int32_t* from_ptr,
      int64_t length);
  Error awkward_cuda_H2D_U32(
      uint32_t** to_ptr,
      uint32_t* from_ptr,
      int64_t length);
  Error awkward_cuda_H2D_64(
    int64_t** to_ptr,
    int64_t* from_ptr,
    int64_t length);
  Error awkward_cuda_H2D_U64(
    uint64_t** to_ptr,
    uint64_t* from_ptr,
    int64_t length);
  Error awkward_cuda_H2D_float32(
    float** to_ptr,
    float* from_ptr,
    int64_t length);
  Error awkward_cuda_H2D_float64(
    double** to_ptr,
    double* from_ptr,
    int64_t length);
}


#endif //AWKWARD_ALLOCATORS_CUH_
