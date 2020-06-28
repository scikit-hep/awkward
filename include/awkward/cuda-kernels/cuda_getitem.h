// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GETITEM_CUH_
#define AWKWARD_GETITEM_CUH_

#include <stdint.h>

extern "C" {
  int8_t awkward_cuda_index8_getitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at);
  uint8_t awkward_cuda_indexU8_getitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at);
  int32_t awkward_cuda_index32_getitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at);
  uint32_t awkward_cuda_indexU32_getitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at);
  int64_t awkward_cuda_index64_getitem_at_nowrap(const int64_t * ptr, int64_t offset, int64_t at);

  bool awkward_cuda_numpyarraybool_getitem_at(
    const bool* ptr,
    int64_t at);
  int8_t awkward_cuda_numpyarray8_getitem_at(
    const int8_t* ptr,
    int64_t at);
  uint8_t awkward_cuda_numpyarrayU8_getitem_at(
    const uint8_t* ptr,
    int64_t at);
  int16_t awkward_cuda_numpyarray16_getitem_at(
    const int16_t* ptr,
    int64_t at);
  uint16_t awkward_cuda_numpyarrayU16_getitem_at(
    const uint16_t* ptr,
    int64_t at);
  int32_t awkward_cuda_numpyarray32_getitem_at(
    const int32_t* ptr,
    int64_t at);
  uint32_t awkward_cuda_numpyarrayU32_getitem_at(
    const uint32_t* ptr,
    int64_t at);
  int64_t awkward_cuda_numpyarray64_getitem_at(
    const int64_t* ptr,
    int64_t at);
  uint64_t awkward_cuda_numpyarrayU64_getitem_at(
    const uint64_t* ptr,
    int64_t at);
  float awkward_cuda_numpyarrayfloat32_getitem_at(
    const float* ptr,
    int64_t at);
  double awkward_cuda_numpyarrayfloat64_getitem_at(
    const double* ptr,
    int64_t at);


  void awkward_cuda_index8_setitem_at_nowrap(const int8_t* ptr, int64_t offset, int64_t at, int8_t value);
  void awkward_cuda_indexU8_setitem_at_nowrap(const uint8_t* ptr, int64_t offset, int64_t at, uint8_t value);
  void awkward_cuda_index32_setitem_at_nowrap(const int32_t* ptr, int64_t offset, int64_t at, int32_t value);
  void awkward_cuda_indexU32_setitem_at_nowrap(const uint32_t* ptr, int64_t offset, int64_t at, uint32_t value);
  void awkward_cuda_index64_setitem_at_nowrap(const int64_t* ptr, int64_t offset, int64_t at, int64_t value);
};

#endif //AWKWARD_GETITEM_CUH_
