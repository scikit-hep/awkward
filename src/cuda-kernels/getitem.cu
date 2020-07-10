// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/cuda-kernels/cuda_getitem.h"
#include <iostream>

template <typename T>
T awkward_cuda_Index_getitem_at_nowrap(const T* ptr,
                                       int64_t offset,
                                       int64_t at) {
  T item;
  cudaMemcpy(&item,
             &ptr[(int64_t) (offset + at)],
             sizeof(T),
             cudaMemcpyDeviceToHost);
  return item;
}
int8_t awkward_cuda_Index8_getitem_at_nowrap(
  const int8_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_Index_getitem_at_nowrap<int8_t>(
    ptr,
    offset,
    at);
}
uint8_t awkward_cuda_IndexU8_getitem_at_nowrap(
  const uint8_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_Index_getitem_at_nowrap<uint8_t>(
    ptr,
    offset,
    at);
}
int32_t awkward_cuda_Index32_getitem_at_nowrap(
  const int32_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_Index_getitem_at_nowrap<int32_t>(
    ptr,
    offset,
    at);
}
uint32_t awkward_cuda_IndexU32_getitem_at_nowrap(
  const uint32_t* ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_Index_getitem_at_nowrap<uint32_t>(
    ptr,
    offset,
    at);
}
int64_t awkward_cuda_Index64_getitem_at_nowrap(
  const int64_t * ptr,
  int64_t offset,
  int64_t at) {
  return awkward_cuda_Index_getitem_at_nowrap<int64_t>(
    ptr,
    offset,
    at);
}

template <typename T>
T awkward_cuda_NumpyArray_getitem_at(const T* ptr,
                                     int64_t at) {
  T item;
  cudaMemcpy(&item,
             &ptr[(int64_t) (at)],
             sizeof(T),
             cudaMemcpyDeviceToHost);
  return item;
}
bool awkward_cuda_NumpyArraybool_getitem_at(
  const bool* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<bool>(
    ptr,
    at);
}
int8_t awkward_cuda_NumpyArray8_getitem_at(
  const int8_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<int8_t>(
    ptr,
    at);
}
uint8_t awkward_cuda_NumpyArrayU8_getitem_at(
  const uint8_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<uint8_t>(
    ptr,
    at);
}
int16_t awkward_cuda_NumpyArray16_getitem_at(
  const int16_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<int16_t>(
    ptr,
    at);
}
uint16_t awkward_cuda_NumpyArrayU16_getitem_at(
  const uint16_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<uint16_t>(
    ptr,
    at);
}
int32_t awkward_cuda_NumpyArray32_getitem_at(
  const int32_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<int32_t>(
    ptr,
    at);
}
uint32_t awkward_cuda_NumpyArrayU32_getitem_at(
  const uint32_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<uint32_t>(
    ptr,
    at);
}
int64_t awkward_cuda_NumpyArray64_getitem_at(
  const int64_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<int64_t>(
    ptr,
    at);
}
uint64_t awkward_cuda_NumpyArrayU64_getitem_at(
  const uint64_t* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<uint64_t>(
    ptr,
    at);
}
float awkward_cuda_NumpyArrayfloat32_getitem_at(
  const float* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<float>(
    ptr,
    at);
}
double awkward_cuda_NumpyArrayfloat64_getitem_at(
  const double* ptr,
  int64_t at) {
  return awkward_cuda_NumpyArray_getitem_at<double>(
    ptr,
    at);
}

template <typename T>
void awkward_cuda_Index_setitem_at_nowrap(
  const T* ptr,
  int64_t offset,
  int64_t at,
  T value) {
  cudaMemcpy(
    (void *) &ptr[(int64_t) (offset + at)],
    &value, sizeof(T),
    cudaMemcpyHostToDevice);
}
void awkward_cuda_Index8_setitem_at_nowrap(
  const int8_t* ptr,
  int64_t offset,
  int64_t at,
  int8_t value) {
  return awkward_cuda_Index_setitem_at_nowrap<int8_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_IndexU8_setitem_at_nowrap(
  const uint8_t* ptr,
  int64_t offset,
  int64_t at,
  uint8_t value) {
  return awkward_cuda_Index_setitem_at_nowrap<uint8_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_Index32_setitem_at_nowrap(
  const int32_t* ptr,
  int64_t offset,
  int64_t at,
  int32_t value) {
  return awkward_cuda_Index_setitem_at_nowrap<int32_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_IndexU32_setitem_at_nowrap(
  const uint32_t* ptr,
  int64_t offset,
  int64_t at,
  uint32_t value) {
  return awkward_cuda_Index_setitem_at_nowrap<uint32_t>(
    ptr,
    offset,
    at,
    value);
}
void awkward_cuda_Index64_setitem_at_nowrap(
  const int64_t* ptr,
  int64_t offset,
  int64_t at,
  int64_t value) {
  return awkward_cuda_Index_setitem_at_nowrap<int64_t>(
    ptr,
    offset,
    at,
    value);
}
