// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H_
#define AWKWARD_KERNEL_UTILS_H_

#include "common.h"

extern "C" {
  EXPORT_SYMBOL int8_t
    awkward_Index8_getitem_at_nowrap(
      const int8_t* ptr,
      int64_t at);
  EXPORT_SYMBOL uint8_t
    awkward_IndexU8_getitem_at_nowrap(
      const uint8_t* ptr,
      int64_t at);
  EXPORT_SYMBOL int32_t
    awkward_Index32_getitem_at_nowrap(
      const int32_t* ptr,
      int64_t at);
  EXPORT_SYMBOL uint32_t
    awkward_IndexU32_getitem_at_nowrap(
      const uint32_t* ptr,
      int64_t at);
  EXPORT_SYMBOL int64_t
    awkward_Index64_getitem_at_nowrap(
      const int64_t* ptr,
      int64_t at);

  EXPORT_SYMBOL void
    awkward_Index8_setitem_at_nowrap(
      int8_t* ptr,
      int64_t at,
      int8_t value);
  EXPORT_SYMBOL void
    awkward_IndexU8_setitem_at_nowrap(
      uint8_t* ptr,
      int64_t at,
      uint8_t value);
  EXPORT_SYMBOL void
    awkward_Index32_setitem_at_nowrap(
      int32_t* ptr,
      int64_t at,
      int32_t value);
  EXPORT_SYMBOL void
    awkward_IndexU32_setitem_at_nowrap(
      uint32_t* ptr,
      int64_t at,
      uint32_t value);
  EXPORT_SYMBOL void
    awkward_Index64_setitem_at_nowrap(
      int64_t* ptr,
      int64_t at,
      int64_t value);

  EXPORT_SYMBOL bool
    awkward_NumpyArraybool_getitem_at0(
      const bool* ptr);
  EXPORT_SYMBOL int8_t
    awkward_NumpyArray8_getitem_at0(
      const int8_t* ptr);
  EXPORT_SYMBOL uint8_t
    awkward_NumpyArrayU8_getitem_at0(
      const uint8_t* ptr);
  EXPORT_SYMBOL int16_t
    awkward_NumpyArray16_getitem_at0(
      const int16_t* ptr);
  EXPORT_SYMBOL uint16_t
    awkward_NumpyArrayU16_getitem_at0(
      const uint16_t* ptr);
  EXPORT_SYMBOL int32_t
    awkward_NumpyArray32_getitem_at0(
      const int32_t* ptr);
  EXPORT_SYMBOL uint32_t
    awkward_NumpyArrayU32_getitem_at0(
      const uint32_t* ptr);
  EXPORT_SYMBOL int64_t
    awkward_NumpyArray64_getitem_at0(
      const int64_t* ptr);
  EXPORT_SYMBOL uint64_t
    awkward_NumpyArrayU64_getitem_at0(
      const uint64_t* ptr);
  EXPORT_SYMBOL float
    awkward_NumpyArrayfloat32_getitem_at0(
      const float* ptr);
  EXPORT_SYMBOL double
    awkward_NumpyArrayfloat64_getitem_at0(
      const double* ptr);

  EXPORT_SYMBOL void
    awkward_regularize_rangeslice(
      int64_t* start,
      int64_t* stop,
      bool posstep,
      bool hasstart,
      bool hasstop,
      int64_t length
    );

}

#endif // AWKWARD_KERNEL_UTILS_H_
