// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_CUDA_ALLOCATORS_H_
#define AWKWARD_CUDA_ALLOCATORS_H_

#include <stdint.h>
#include "awkward/common.h"

extern "C" {
  int awkward_cuda_ptr_loc(void* ptr);

  EXPORT_SYMBOL bool *awkward_cuda_ptrbool_alloc(int64_t length);
  EXPORT_SYMBOL int8_t *awkward_cuda_ptr8_alloc(int64_t length);
  EXPORT_SYMBOL  uint8_t *awkward_cuda_ptrU8_alloc(int64_t length);
  EXPORT_SYMBOL int16_t *awkward_cuda_ptr16_alloc(int64_t length);
  EXPORT_SYMBOL uint16_t *awkward_cuda_ptrU16_alloc(int64_t length);
  EXPORT_SYMBOL int32_t *awkward_cuda_ptr32_alloc(int64_t length);
  EXPORT_SYMBOL uint32_t *awkward_cuda_ptrU32_alloc(int64_t length);
  EXPORT_SYMBOL int64_t *awkward_cuda_ptr64_alloc(int64_t length);
  EXPORT_SYMBOL uint64_t *awkward_cuda_ptrU64_alloc(int64_t length);
  EXPORT_SYMBOL float *awkward_cuda_ptrfloat32_alloc(int64_t length);
  EXPORT_SYMBOL double *awkward_cuda_ptrfloat64_alloc(int64_t length);

  EXPORT_SYMBOL ERROR awkward_cuda_ptrbool_dealloc(const bool* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptr8_dealloc(const int8_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptrU8_dealloc(const uint8_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptr16_dealloc(const int16_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptrU16_dealloc(const uint16_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptr32_dealloc(const int32_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptrU32_dealloc(const uint32_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptr64_dealloc(const int64_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptrU64_dealloc(const uint64_t* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptrfloat32_dealloc(const float* ptr);
  EXPORT_SYMBOL ERROR awkward_cuda_ptrfloat64_dealloc(const double* ptr);

  EXPORT_SYMBOL ERROR awkward_cuda_H2Dbool(
      bool** to_ptr,
      bool* from_ptr,
      int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2D8(
      int8_t** to_ptr,
      int8_t* from_ptr,
      int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2DU8(
        uint8_t** to_ptr,
        uint8_t* from_ptr,
        int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2D16(
        int16_t** to_ptr,
        int16_t* from_ptr,
        int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2DU16(
        uint16_t** to_ptr,
        uint16_t* from_ptr,
        int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2D32(
        int32_t** to_ptr,
        int32_t* from_ptr,
        int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2DU32(
        uint32_t** to_ptr,
        uint32_t* from_ptr,
        int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2D64(
      int64_t** to_ptr,
      int64_t* from_ptr,
      int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2DU64(
      uint64_t** to_ptr,
      uint64_t* from_ptr,
      int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2Dfloat32(
      float** to_ptr,
      float* from_ptr,
      int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_H2Dfloat64(
      double** to_ptr,
      double* from_ptr,
      int64_t length);

  EXPORT_SYMBOL ERROR awkward_cuda_D2Hbool(
    bool** to_ptr,
    bool* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2H8(
    int8_t** to_ptr,
    int8_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2HU8(
    uint8_t** to_ptr,
    uint8_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2H16(
    int16_t** to_ptr,
    int16_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2HU16(
    uint16_t** to_ptr,
    uint16_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2H32(
    int32_t** to_ptr,
    int32_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2HU32(
    uint32_t** to_ptr,
    uint32_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2H64(
    int64_t** to_ptr,
    int64_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2HU64(
    uint64_t** to_ptr,
    uint64_t* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2Hfloat32(
    float** to_ptr,
    float* from_ptr,
    int64_t length);
  EXPORT_SYMBOL ERROR awkward_cuda_D2Hfloat64(
    double** to_ptr,
    double* from_ptr,
    int64_t length);
}


#endif //AWKWARD_CUDA_ALLOCATORS_H_
