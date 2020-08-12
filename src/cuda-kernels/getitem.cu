// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/kernels/getitem.h"
#include <iostream>

template <typename T>
T awkward_Index_getitem_at_nowrap(const T* ptr, int64_t at) {
  T item;
  cudaMemcpy(&item, &ptr[at], sizeof(T), cudaMemcpyDeviceToHost);
  return item;
}
int8_t awkward_Index8_getitem_at_nowrap(
  const int8_t* ptr,
  int64_t at) {
  return awkward_Index_getitem_at_nowrap<int8_t>(
    ptr,
    at);
}
uint8_t awkward_IndexU8_getitem_at_nowrap(
  const uint8_t* ptr,
  int64_t at) {
  return awkward_Index_getitem_at_nowrap<uint8_t>(
    ptr,
    at);
}
int32_t awkward_Index32_getitem_at_nowrap(
  const int32_t* ptr,
  int64_t at) {
  return awkward_Index_getitem_at_nowrap<int32_t>(
    ptr,
    at);
}
uint32_t awkward_IndexU32_getitem_at_nowrap(
  const uint32_t* ptr,
  int64_t at) {
  return awkward_Index_getitem_at_nowrap<uint32_t>(
    ptr,
    at);
}
int64_t awkward_Index64_getitem_at_nowrap(
  const int64_t * ptr,
  int64_t at) {
  return awkward_Index_getitem_at_nowrap<int64_t>(
    ptr,
    at);
}

template <typename T>
T awkward_NumpyArray_getitem_at0(const T* ptr) {
  T item;
  cudaMemcpy(&item,
             &ptr[0],
             sizeof(T),
             cudaMemcpyDeviceToHost);
  return item;
}
bool awkward_NumpyArraybool_getitem_at0(
  const bool* ptr) {
  return awkward_NumpyArray_getitem_at0<bool>(ptr);
}
int8_t awkward_NumpyArray8_getitem_at0(
  const int8_t* ptr) {
  return awkward_NumpyArray_getitem_at0<int8_t>(ptr);
}
uint8_t awkward_NumpyArrayU8_getitem_at0(
  const uint8_t* ptr) {
  return awkward_NumpyArray_getitem_at0<uint8_t>(ptr);
}
int16_t awkward_NumpyArray16_getitem_at0(
  const int16_t* ptr) {
  return awkward_NumpyArray_getitem_at0<int16_t>(ptr);
}
uint16_t awkward_NumpyArrayU16_getitem_at0(
  const uint16_t* ptr) {
  return awkward_NumpyArray_getitem_at0<uint16_t>(ptr);
}
int32_t awkward_NumpyArray32_getitem_at0(
  const int32_t* ptr) {
  return awkward_NumpyArray_getitem_at0<int32_t>(ptr);
}
uint32_t awkward_NumpyArrayU32_getitem_at0(
  const uint32_t* ptr) {
  return awkward_NumpyArray_getitem_at0<uint32_t>(ptr);
}
int64_t awkward_NumpyArray64_getitem_at0(
  const int64_t* ptr) {
  return awkward_NumpyArray_getitem_at0<int64_t>(ptr);
}
uint64_t awkward_NumpyArrayU64_getitem_at0(
  const uint64_t* ptr) {
  return awkward_NumpyArray_getitem_at0<uint64_t>(ptr);
}
float awkward_NumpyArrayfloat32_getitem_at0(
  const float* ptr) {
  return awkward_NumpyArray_getitem_at0<float>(ptr);
}
double awkward_NumpyArrayfloat64_getitem_at0(
  const double* ptr) {
  return awkward_NumpyArray_getitem_at0<double>(ptr);
}

template <typename T>
void awkward_Index_setitem_at_nowrap(
  const T* ptr,
  int64_t at,
  T value) {
  cudaMemcpy((void *) &ptr[at], &value, sizeof(T), cudaMemcpyHostToDevice);
}
void awkward_Index8_setitem_at_nowrap(
  const int8_t* ptr,
  int64_t at,
  int8_t value) {
  return awkward_Index_setitem_at_nowrap<int8_t>(
    ptr,
    at,
    value);
}
void awkward_IndexU8_setitem_at_nowrap(
  const uint8_t* ptr,
  int64_t at,
  uint8_t value) {
  return awkward_Index_setitem_at_nowrap<uint8_t>(
    ptr,
    at,
    value);
}
void awkward_Index32_setitem_at_nowrap(
  const int32_t* ptr,
  int64_t at,
  int32_t value) {
  return awkward_Index_setitem_at_nowrap<int32_t>(
    ptr,
    at,
    value);
}
void awkward_IndexU32_setitem_at_nowrap(
  const uint32_t* ptr,
  int64_t at,
  uint32_t value) {
  return awkward_Index_setitem_at_nowrap<uint32_t>(
    ptr,
    at,
    value);
}
void awkward_Index64_setitem_at_nowrap(
  const int64_t* ptr,
  int64_t at,
  int64_t value) {
  return awkward_Index_setitem_at_nowrap<int64_t>(
    ptr,
    at,
    value);
}
