// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNELS_ALLOCATORS_H_
#define AWKWARD_KERNELS_ALLOCATORS_H_

#include <cstdint>
#include <stdlib.h>
#include "awkward/common.h"

extern "C" {
  EXPORT_SYMBOL int8_t *awkward_ptr8_alloc(int64_t length);
  EXPORT_SYMBOL uint8_t *awkward_ptrU8_alloc(int64_t length);
  EXPORT_SYMBOL int16_t *awkward_ptr16_alloc(int64_t length);
  EXPORT_SYMBOL uint16_t *awkward_ptrU16_alloc(int64_t length);
  EXPORT_SYMBOL int32_t *awkward_ptr32_alloc(int64_t length);
  EXPORT_SYMBOL uint32_t *awkward_ptrU32_alloc(int64_t length);
  EXPORT_SYMBOL int64_t *awkward_ptr64_alloc(int64_t length);
  EXPORT_SYMBOL uint64_t *awkward_ptrU64_alloc(int64_t length);
  EXPORT_SYMBOL float *awkward_ptrfloat32_alloc(int64_t length);
  EXPORT_SYMBOL double *awkward_ptrfloat64_alloc(int64_t length);
  EXPORT_SYMBOL bool *awkward_ptrbool_alloc(int64_t length);

  EXPORT_SYMBOL ERROR awkward_ptrbool_dealloc(const bool* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrchar_dealloc(const char* ptr);
  EXPORT_SYMBOL ERROR awkward_ptr8_dealloc(const int8_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrU8_dealloc(const uint8_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptr16_dealloc(const int16_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrU16_dealloc(const uint16_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptr32_dealloc(const int32_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrU32_dealloc(const uint32_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptr64_dealloc(const int64_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrU64_dealloc(const uint64_t* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrfloat32_dealloc(const float* ptr);
  EXPORT_SYMBOL ERROR awkward_ptrfloat64_dealloc(const double* ptr);
};

#endif //AWKWARD_KERNELS_ALLOCATORS_H_
