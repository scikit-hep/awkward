// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/cuda_getitem.h"
#include <iostream>

template <typename T>
T awkward_cuda_index_getitem_at_nowrap(const T* ptr,
                                       int64_t offset,
                                       int64_t at) {
  T item;
  cudaMemcpy(&item,
             &ptr[(int64_t) (offset + at)],
             sizeof(T),
             cudaMemcpyDeviceToHost);
  return item;
}
int8_t awkward_cuda_index8_getitem_at_nowrap(
  const int8_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_index_getitem_at_nowrap<int8_t>(
    ptr,
    offset,
    at);
}
uint8_t awkward_cuda_indexU8_getitem_at_nowrap(
  const uint8_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_index_getitem_at_nowrap<uint8_t>(
    ptr,
    offset,
    at);
}
int32_t awkward_cuda_index32_getitem_at_nowrap(
  const int32_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_index_getitem_at_nowrap<int32_t>(
    ptr,
    offset,
    at);
}
uint32_t awkward_cuda_indexU32_getitem_at_nowrap(
  const uint32_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_index_getitem_at_nowrap<uint32_t>(
    ptr,
    offset,
    at);
}
int64_t awkward_cuda_index64_getitem_at_nowrap(
  const int64_t * ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_index_getitem_at_nowrap<int64_t>(
    ptr,
    offset,
    at);
}

template <typename T>
void awkward_cuda_index_setitem_at_nowrap(
  const T* ptr,
  int64_t offset,
  int64_t at,
  T value) {
  cudaMemcpy(
    (void *) &ptr[(int64_t) (offset + at)],
    &value, sizeof(T),
    cudaMemcpyHostToDevice);
}
void awkward_cuda_index8_setitem_at_nowrap(
  const int8_t* ptr,
  int64_t offset,
  int64_t at,
  int8_t value) {
  return awkward_cuda_index_setitem_at_nowrap<int8_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_indexU8_setitem_at_nowrap(
  const uint8_t* ptr,
  int64_t offset,
  int64_t at,
  uint8_t value) {
  return awkward_cuda_index_setitem_at_nowrap<uint8_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_index32_setitem_at_nowrap(
  const int32_t* ptr,
  int64_t offset,
  int64_t at,
  int32_t value) {
  return awkward_cuda_index_setitem_at_nowrap<int32_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_indexU32_setitem_at_nowrap(
  const uint32_t* ptr,
  int64_t offset,
  int64_t at,
  uint32_t value) {
  return awkward_cuda_index_setitem_at_nowrap<uint32_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_index64_setitem_at_nowrap(
  const int64_t* ptr,
  int64_t offset,
  int64_t at,
  int64_t value) {
  return awkward_cuda_index_setitem_at_nowrap<int64_t>(
    ptr,
    offset,
    at,
    value);
}
