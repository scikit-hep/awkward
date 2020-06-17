// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/allocators.h"


template <typename T>
T *awkward_cpu_ptr_alloc(int64_t length) {
  if(length != 0) {
    return new T[(size_t) length];
  }
  return nullptr;
}
bool *awkward_cpu_ptrbool_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<bool>(length);
}
int8_t *awkward_cpu_ptr8_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<int8_t>(length);
}
uint8_t *awkward_cpu_ptrU8_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<uint8_t>(length);
}
int16_t *awkward_cpu_ptr16_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<int16_t>(length);
}
uint16_t *awkward_cpu_ptrU16_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<uint16_t>(length);
}
int32_t *awkward_cpu_ptr32_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<int32_t>(length);
}
uint32_t *awkward_cpu_ptrU32_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<uint32_t>(length);
}
int64_t *awkward_cpu_ptr64_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<int64_t>(length);
}
uint64_t *awkward_cpu_ptrU64_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<uint64_t>(length);
}
float *awkward_cpu_ptrfloat32_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<float>(length);
}
double *awkward_cpu_ptrfloat64_alloc(int64_t length) {
  return awkward_cpu_ptr_alloc<double>(length);
}
