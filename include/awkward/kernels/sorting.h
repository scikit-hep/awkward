// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNELS_SORTING_H_
#define AWKWARD_KERNELS_SORTING_H_

#include "awkward/common.h"

extern "C" {
  /// @param toindex outparam
  /// @param tolength inparam
  /// @param parents inparam role: IndexedArray-index
  /// @param parentslength inparam
  EXPORT_SYMBOL struct Error
  awkward_sorting_ranges(
    int64_t* toindex,
    int64_t tolength,
    const int64_t* parents,
    int64_t parentslength);

  /// @param tolength outparam role: pointer
  /// @param parents inparam role: IndexedArray-index
  /// @param parentslength inparam
  EXPORT_SYMBOL struct Error
  awkward_sorting_ranges_length(
    int64_t* tolength,
    const int64_t* parents,
    int64_t parentslength);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_bool(
    int64_t* toptr,
    const bool* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_int8(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_uint8(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_int16(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_uint16(
    int64_t* toptr,
    const uint16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_int32(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_uint32(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_int64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_uint64(
    int64_t* toptr,
    const uint64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_float32(
    int64_t* toptr,
    const float* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
  awkward_argsort_float64(
    int64_t* toptr,
    const double* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_bool(
      bool* toptr,
      const bool* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_int8(
      int8_t* toptr,
      const int8_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_uint8(
      uint8_t* toptr,
      const uint8_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_int16(
      int16_t* toptr,
      const int16_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_uint16(
      uint16_t* toptr,
      const uint16_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_int32(
      int32_t* toptr,
      const int32_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_uint32(
      uint32_t* toptr,
      const uint32_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_int64(
      int64_t* toptr,
      const int64_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_uint64(
      uint64_t* toptr,
      const uint64_t* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_float32(
      float* toptr,
      const float* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param parentslength inparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_sort_float64(
      double* toptr,
      const double* fromptr,
      int64_t length,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t parentslength,
      bool ascending,
      bool stable);

  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_local_preparenext_64(
      int64_t* tocarry,
      const int64_t* fromindex,
      int64_t length);

  /// @param tocarry outparam
  /// @param starts inparam role: ListArray-starts
  /// @param parents inparam role: IndexedArray-index
  /// @param parentslength inparam
  /// @param nextparents inparam role: IndexedArray-index
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_local_preparenext_64(
      int64_t* tocarry,
      const int64_t* starts,
      const int64_t* parents,
      int64_t parentslength,
      const int64_t* nextparents);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param outoffsets outparam
  /// @param ascending inparam role: ListArray-replacement
  /// @param stable inparam role: ListArray-replacement
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_sort_asstrings_uint8(
      uint8_t* toptr,
      const uint8_t* fromptr,
      const int64_t* offsets,
      int64_t offsetslength,
      int64_t* outoffsets,
      bool ascending,
      bool stable);

}

#endif // AWKWARD_KERNELS_SORTING_H_
