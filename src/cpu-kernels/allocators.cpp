// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/allocators.h"


template <typename T>
T *awkward_ptr_alloc(int64_t length) {
  if(length != 0) {
    return new T[(size_t) length];
  }
  return nullptr;
}
bool *awkward_ptrbool_alloc(int64_t length) {
  return awkward_ptr_alloc<bool>(length);
}
int8_t *awkward_ptr8_alloc(int64_t length) {
  return awkward_ptr_alloc<int8_t>(length);
}
uint8_t *awkward_ptrU8_alloc(int64_t length) {
  return awkward_ptr_alloc<uint8_t>(length);
}
int16_t *awkward_ptr16_alloc(int64_t length) {
  return awkward_ptr_alloc<int16_t>(length);
}
uint16_t *awkward_ptrU16_alloc(int64_t length) {
  return awkward_ptr_alloc<uint16_t>(length);
}
int32_t *awkward_ptr32_alloc(int64_t length) {
  return awkward_ptr_alloc<int32_t>(length);
}
uint32_t *awkward_ptrU32_alloc(int64_t length) {
  return awkward_ptr_alloc<uint32_t>(length);
}
int64_t *awkward_ptr64_alloc(int64_t length) {
  return awkward_ptr_alloc<int64_t>(length);
}
uint64_t *awkward_ptrU64_alloc(int64_t length) {
  return awkward_ptr_alloc<uint64_t>(length);
}
float *awkward_ptrfloat32_alloc(int64_t length) {
  return awkward_ptr_alloc<float>(length);
}
double *awkward_ptrfloat64_alloc(int64_t length) {
  return awkward_ptr_alloc<double>(length);
}

template <typename  T>
ERROR awkward_ptr_dealloc(const T* ptr) {
 delete[] ptr;
 return success();
}
ERROR awkward_ptrbool_dealloc(const bool *ptr) {
  return awkward_ptr_dealloc<bool>(ptr);
}
ERROR awkward_ptrchar_dealloc(const char *ptr) {
  return awkward_ptr_dealloc<char>(ptr);
}
ERROR awkward_ptr8_dealloc(const int8_t *ptr) {
  return awkward_ptr_dealloc<int8_t>(ptr);
}
ERROR awkward_ptrU8_dealloc(const uint8_t *ptr) {
  return awkward_ptr_dealloc<uint8_t>(ptr);
}
ERROR awkward_ptr16_dealloc(const int16_t *ptr) {
  return awkward_ptr_dealloc<int16_t>(ptr);
}
ERROR awkward_ptrU16_dealloc(const uint16_t *ptr) {
  return awkward_ptr_dealloc<uint16_t>(ptr);
}
ERROR awkward_ptr32_dealloc(const int32_t *ptr) {
  return awkward_ptr_dealloc<int32_t>(ptr);
}
ERROR awkward_ptrU32_dealloc(const uint32_t *ptr) {
  return awkward_ptr_dealloc<uint32_t>(ptr);
}
ERROR awkward_ptr64_dealloc(const int64_t *ptr) {
  return awkward_ptr_dealloc<int64_t>(ptr);
}
ERROR awkward_ptrU64_dealloc(const uint64_t *ptr) {
  return awkward_ptr_dealloc<uint64_t>(ptr);
}
ERROR awkward_ptrfloat32_dealloc(const float *ptr) {
  return awkward_ptr_dealloc<float>(ptr);
}
ERROR awkward_ptrfloat64_dealloc(const double *ptr) {
  return awkward_ptr_dealloc<double>(ptr);
}
