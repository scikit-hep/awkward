// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GETITEM_H
#define AWKWARD_GETITEM_H

#include <stdint.h>

extern "C" {
  int8_t awkward_cuda_Index8_getitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at);
  uint8_t awkward_cuda_IndexU8_getitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at);
  int32_t awkward_cuda_Index32_getitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at);
  uint32_t awkward_cuda_IndexU32_getitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at);
  int64_t awkward_cuda_Index64_getitem_at_nowrap(const int64_t * ptr, int64_t offset, int64_t at);

  bool awkward_cuda_NumpyArraybool_getitem_at(
    const bool* ptr,
    int64_t at);
  int8_t awkward_cuda_NumpyArray8_getitem_at(
    const int8_t* ptr,
    int64_t at);
  uint8_t awkward_cuda_NumpyArrayU8_getitem_at(
    const uint8_t* ptr,
    int64_t at);
  int16_t awkward_cuda_NumpyArray16_getitem_at(
    const int16_t* ptr,
    int64_t at);
  uint16_t awkward_cuda_NumpyArrayU16_getitem_at(
    const uint16_t* ptr,
    int64_t at);
  int32_t awkward_cuda_NumpyArray32_getitem_at(
    const int32_t* ptr,
    int64_t at);
  uint32_t awkward_cuda_NumpyArrayU32_getitem_at(
    const uint32_t* ptr,
    int64_t at);
  int64_t awkward_cuda_NumpyArray64_getitem_at(
    const int64_t* ptr,
    int64_t at);
  uint64_t awkward_cuda_NumpyArrayU64_getitem_at(
    const uint64_t* ptr,
    int64_t at);
  float awkward_cuda_NumpyArrayfloat32_getitem_at(
    const float* ptr,
    int64_t at);
  double awkward_cuda_NumpyArrayfloat64_getitem_at(
    const double* ptr,
    int64_t at);


  void awkward_cuda_Index8_setitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at, int8_t value);
  void awkward_cuda_IndexU8_setitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at, uint8_t value);
  void awkward_cuda_Index32_setitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at, int32_t value);
  void awkward_cuda_IndexU32_setitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at, uint32_t value);
  void awkward_cuda_Index64_setitem_at_nowrap(const int64_t* ptr, int64_t offset, int64_t at, int64_t value);
};

#endif //AWKWARD_GETITEM_H
