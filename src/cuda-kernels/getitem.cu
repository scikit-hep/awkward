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
T awkward_cuda_numpyarray_getitem_at(const T* ptr,
                                     int64_t at) {
  T item;
  cudaMemcpy(&item,
             &ptr[(int64_t) (at)],
             sizeof(T),
             cudaMemcpyDeviceToHost);
  return item;
}
bool awkward_cuda_numpyarraybool_getitem_at(
  const bool* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<bool>(
    ptr,
    at);
}
int8_t awkward_cuda_numpyarray8_getitem_at(
  const int8_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<int8_t>(
    ptr,
    at);
}
uint8_t awkward_cuda_numpyarrayU8_getitem_at(
  const uint8_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<uint8_t>(
    ptr,
    at);
}
int16_t awkward_cuda_numpyarray16_getitem_at(
  const int16_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<int16_t>(
    ptr,
    at);
}
uint16_t awkward_cuda_numpyarrayU16_getitem_at(
  const uint16_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<uint16_t>(
    ptr,
    at);
}
int32_t awkward_cuda_numpyarray32_getitem_at(
  const int32_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<int32_t>(
    ptr,
    at);
}
uint32_t awkward_cuda_numpyarrayU32_getitem_at(
  const uint32_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<uint32_t>(
    ptr,
    at);
}
int64_t awkward_cuda_numpyarray64_getitem_at(
  const int64_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<int64_t>(
    ptr,
    at);
}
uint64_t awkward_cuda_numpyarrayU64_getitem_at(
  const uint64_t* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<uint64_t>(
    ptr,
    at);
}
float awkward_cuda_numpyarrayfloat32_getitem_at(
  const float* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<float>(
    ptr,
    at);
}
double awkward_cuda_numpyarrayfloat64_getitem_at(
  const double* ptr,
  int64_t at) {
  return awkward_cuda_numpyarray_getitem_at<double>(
    ptr,
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
