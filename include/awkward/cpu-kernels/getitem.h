// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_GETITEM_H_
#define AWKWARDCPU_GETITEM_H_

#include "awkward/common.h"

extern "C" {
  /// @param start inoutparam role: pointer
  /// @param stop inoutparam role: pointer
  /// @param posstep inparam role: ListArray-replacement
  /// @param hasstart inparam role: ListArray-replacement
  /// @param hasstop inparam role: ListArray-replacement
  /// @param length inparam
  EXPORT_SYMBOL void
    awkward_regularize_rangeslice(
      int64_t* start,
      int64_t* stop,
      bool posstep,
      bool hasstart,
      bool hasstop,
      int64_t length);

  /// @param flatheadptr inparam role: NumpyArray-ptr
  /// @param lenflathead inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_regularize_arrayslice_64(
      int64_t* flatheadptr,
      int64_t lenflathead,
      int64_t length);

  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index8_to_Index64(
      int64_t* toptr,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexU8_to_Index64(
      int64_t* toptr,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index32_to_Index64(
      int64_t* toptr,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param fromptr inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexU32_to_Index64(
      int64_t* toptr,
      const uint32_t* fromptr,
      int64_t length);

  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param lenfromindex inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index8_carry_64(
      int8_t* toindex,
      const int8_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t lenfromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param lenfromindex inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexU8_carry_64(
      uint8_t* toindex,
      const uint8_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t lenfromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param lenfromindex inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index32_carry_64(
      int32_t* toindex,
      const int32_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t lenfromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param lenfromindex inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexU32_carry_64(
      uint32_t* toindex,
      const uint32_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t lenfromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param lenfromindex inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index64_carry_64(
      int64_t* toindex,
      const int64_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t lenfromindex,
      int64_t length);

  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index8_carry_nocheck_64(
      int8_t* toindex,
      const int8_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexU8_carry_nocheck_64(
      uint8_t* toindex,
      const uint8_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index32_carry_nocheck_64(
      int32_t* toindex,
      const int32_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexU32_carry_nocheck_64(
      uint32_t* toindex,
      const uint32_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 1
  /// @param carry inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 1
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Index64_carry_nocheck_64(
      int64_t* toindex,
      const int64_t* fromindex,
      const int64_t* carry,
      int64_t fromindexoffset,
      int64_t length);

  /// @param toptr outparam
  /// @param fromptr inparam
  /// @param ndim inparam
  /// @param shape inparam
  /// @param strides inparam
  EXPORT_SYMBOL struct Error
    awkward_slicearray_ravel_64(
      int64_t* toptr,
      const int64_t* fromptr,
      int64_t ndim,
      const int64_t* shape,
      const int64_t* strides);

  /// @param same outparam role: pointer
  /// @param bytemask inparam role: ByteMaskedArray-mask instance: 0
  /// @param bytemaskoffset inparam role: ByteMaskedArray-mask-offset instance: 1
  /// @param missingindex inparam role: ByteMaskedArray-mask instance: 1
  /// @param missingindexoffset inparam role: ByteMaskedArray-mask-offset instance: 1
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_slicemissing_check_same(
      bool* same,
      const int8_t* bytemask,
      int64_t bytemaskoffset,
      const int64_t* missingindex,
      int64_t missingindexoffset,
      int64_t length);

  /// @param toptr outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_carry_arange32(
      int32_t* toptr,
      int64_t length);
  /// @param toptr outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_carry_arangeU32(
      uint32_t* toptr,
      int64_t length);
  /// @param toptr outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_carry_arange64(
      int64_t* toptr,
      int64_t length);

  /// @param newidentitiesptr outparam
  /// @param identitiesptr inparam role: IndexedArray-index
  /// @param carryptr inparam role: ListArray-at-array
  /// @param lencarry inparam
  /// @param offset inparam role: IndexedArray-index-offset
  /// @param width inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities32_getitem_carry_64(
      int32_t* newidentitiesptr,
      const int32_t* identitiesptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t offset,
      int64_t width,
      int64_t length);
  /// @param newidentitiesptr outparam
  /// @param identitiesptr inparam role: IndexedArray-index
  /// @param carryptr inparam role: ListArray-at-array
  /// @param lencarry inparam
  /// @param offset inparam role: IndexedArray-index-offset
  /// @param width inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_Identities64_getitem_carry_64(
      int64_t* newidentitiesptr,
      const int64_t* identitiesptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t offset,
      int64_t width,
      int64_t length);

  /// @param toptr outparam
  /// @param skip inparam
  /// @param stride inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_contiguous_init_64(
      int64_t* toptr,
      int64_t skip,
      int64_t stride);

  /// @param toptr outparam
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param len inparam
  /// @param stride inparam
  /// @param offset inparam role: NumpyArray-ptr-offset
  /// @param pos inparam role: IndexedArray-index
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_contiguous_copy_64(
      uint8_t* toptr,
      const uint8_t* fromptr,
      int64_t len,
      int64_t stride,
      int64_t offset,
      const int64_t* pos);

  /// @param topos outparam
  /// @param frompos inparam role: NumpyArray-ptr
  /// @param length inparam
  /// @param skip inparam
  /// @param stride inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_contiguous_next_64(
      int64_t* topos,
      const int64_t* frompos,
      int64_t length,
      int64_t skip,
      int64_t stride);

  /// @param toptr outparam
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param len inparam
  /// @param stride inparam
  /// @param offset inparam role: NumpyArray-ptr-offset
  /// @param pos inparam role: IndexedArray-index
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_next_null_64(
      uint8_t* toptr,
      const uint8_t* fromptr,
      int64_t len,
      int64_t stride,
      int64_t offset,
      const int64_t* pos);

  /// @param nextcarryptr outparam
  /// @param carryptr inparam role: NumpyArray-ptr
  /// @param lencarry inparam
  /// @param skip inparam
  /// @param at inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_next_at_64(
      int64_t* nextcarryptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t skip,
      int64_t at);

  /// @param nextcarryptr outparam
  /// @param carryptr inparam role: NumpyArray-ptr
  /// @param lencarry inparam
  /// @param lenhead inparam
  /// @param skip inparam
  /// @param start inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_next_range_64(
      int64_t* nextcarryptr,
      const int64_t* carryptr,
      int64_t lencarry,
      int64_t lenhead,
      int64_t skip,
      int64_t start,
      int64_t step);

  /// @param nextcarryptr outparam
  /// @param nextadvancedptr outparam
  /// @param carryptr inparam role: NumpyArray-ptr instance: 0
  /// @param advancedptr inparam role: NumpyArray-ptr instance: 1
  /// @param lencarry inparam
  /// @param lenhead inparam
  /// @param skip inparam
  /// @param start inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_next_range_advanced_64(
      int64_t* nextcarryptr,
      int64_t* nextadvancedptr,
      const int64_t* carryptr,
      const int64_t* advancedptr,
      int64_t lencarry,
      int64_t lenhead,
      int64_t skip,
      int64_t start,
      int64_t step);

  /// @param nextcarryptr outparam
  /// @param nextadvancedptr outparam
  /// @param carryptr inparam role: NumpyArray-ptr instance: 0
  /// @param flatheadptr inparam role: NumpyArray-ptr instance: 1
  /// @param lencarry inparam
  /// @param lenflathead inparam
  /// @param skip inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_next_array_64(
      int64_t* nextcarryptr,
      int64_t* nextadvancedptr,
      const int64_t* carryptr,
      const int64_t* flatheadptr,
      int64_t lencarry,
      int64_t lenflathead,
      int64_t skip);

  /// @param nextcarryptr outparam
  /// @param carryptr inparam role: NumpyArray-ptr instance: 0
  /// @param advancedptr inparam role: NumpyArray-ptr instance: 0
  /// @param flatheadptr inparam role: NumpyArray-ptr instance: 1
  /// @param lencarry inparam
  /// @param skip inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_next_array_advanced_64(
      int64_t* nextcarryptr,
      const int64_t* carryptr,
      const int64_t* advancedptr,
      const int64_t* flatheadptr,
      int64_t lencarry,
      int64_t skip);

  /// @param numtrue outparam role: pointer
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param byteoffset inparam role: NumpyArray-ptr-offset
  /// @param length inparam
  /// @param stride inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_boolean_numtrue(
      int64_t* numtrue,
      const int8_t* fromptr,
      int64_t byteoffset,
      int64_t length,
      int64_t stride);
  /// @param toptr outparam role: pointer
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param byteoffset inparam role: NumpyArray-ptr-offset
  /// @param length inparam
  /// @param stride inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_getitem_boolean_nonzero_64(
      int64_t* toptr,
      const int8_t* fromptr,
      int64_t byteoffset,
      int64_t length,
      int64_t stride);

  /// @param tocarry outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lenstarts inparam
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param at inparam role: ListArray-at
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_at_64(
      int64_t* tocarry,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t at);
  /// @param tocarry outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lenstarts inparam
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param at inparam role: ListArray-at
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_at_64(
      int64_t* tocarry,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t at);
  /// @param tocarry outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lenstarts inparam
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param at inparam role: ListArray-at
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_at_64(
      int64_t* tocarry,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t at);

  /// @param carrylength outparam
  /// @param fromstarts inparam
  /// @param fromstops inparam
  /// @param lenstarts inparam
  /// @param startsoffset inparam
  /// @param stopsoffset inparam
  /// @param start inparam
  /// @param stop inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_range_carrylength(
      int64_t* carrylength,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t start,
      int64_t stop,
      int64_t step);
  /// @param carrylength outparam
  /// @param fromstarts inparam
  /// @param fromstops inparam
  /// @param lenstarts inparam
  /// @param startsoffset inparam
  /// @param stopsoffset inparam
  /// @param start inparam
  /// @param stop inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_range_carrylength(
      int64_t* carrylength,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t start,
      int64_t stop,
      int64_t step);
  /// @param carrylength outparam
  /// @param fromstarts inparam
  /// @param fromstops inparam
  /// @param lenstarts inparam
  /// @param startsoffset inparam
  /// @param stopsoffset inparam
  /// @param start inparam
  /// @param stop inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_range_carrylength(
      int64_t* carrylength,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t start,
      int64_t stop,
      int64_t step);

  /// @param tooffsets outparam
  /// @param tocarry outparam
  /// @param fromstarts inparam
  /// @param fromstops inparam
  /// @param lenstarts inparam
  /// @param startsoffset inparam
  /// @param stopsoffset inparam
  /// @param start inparam
  /// @param stop inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_range_64(
      int32_t* tooffsets,
      int64_t* tocarry,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t start,
      int64_t stop,
      int64_t step);
  /// @param tooffsets outparam
  /// @param tocarry outparam
  /// @param fromstarts inparam
  /// @param fromstops inparam
  /// @param lenstarts inparam
  /// @param startsoffset inparam
  /// @param stopsoffset inparam
  /// @param start inparam
  /// @param stop inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_range_64(
      uint32_t* tooffsets,
      int64_t* tocarry,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t start,
      int64_t stop,
      int64_t step);
  /// @param tooffsets outparam
  /// @param tocarry outparam
  /// @param fromstarts inparam
  /// @param fromstops inparam
  /// @param lenstarts inparam
  /// @param startsoffset inparam
  /// @param stopsoffset inparam
  /// @param start inparam
  /// @param stop inparam
  /// @param step inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_range_64(
      int64_t* tooffsets,
      int64_t* tocarry,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t lenstarts,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t start,
      int64_t stop,
      int64_t step);

  /// @param total outparam role: pointer
  /// @param fromoffsets inparam role: ListArray-starts
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_range_counts_64(
      int64_t* total,
      const int32_t* fromoffsets,
      int64_t lenstarts);
  /// @param total outparam role: pointer
  /// @param fromoffsets inparam role: ListArray-starts
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_range_counts_64(
      int64_t* total,
      const uint32_t* fromoffsets,
      int64_t lenstarts);
  /// @param total outparam role: pointer
  /// @param fromoffsets inparam role: ListArray-starts
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_range_counts_64(
      int64_t* total,
      const int64_t* fromoffsets,
      int64_t lenstarts);

  /// @param toadvanced outparam
  /// @param fromadvanced inparam role: ListArray-starts
  /// @param fromoffsets inparam role: ListArray-diff
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_range_spreadadvanced_64(
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      const int32_t* fromoffsets,
      int64_t lenstarts);
  /// @param toadvanced outparam
  /// @param fromadvanced inparam role: ListArray-starts
  /// @param fromoffsets inparam role: ListArray-diff
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_range_spreadadvanced_64(
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      const uint32_t* fromoffsets,
      int64_t lenstarts);
  /// @param toadvanced outparam
  /// @param fromadvanced inparam role: ListArray-starts
  /// @param fromoffsets inparam role: ListArray-diff
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_range_spreadadvanced_64(
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      const int64_t* fromoffsets,
      int64_t lenstarts);

  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromarray inparam role: ListArray-at-array
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lenarray inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_array_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      const int64_t* fromarray,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromarray inparam role: ListArray-at-array
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lenarray inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_array_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      const int64_t* fromarray,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromarray inparam role: ListArray-at-array
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lenarray inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_array_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      const int64_t* fromarray,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);

  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromarray inparam role: ListArray-at-array
  /// @param fromadvanced inparam role: ListArray-diff
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lenarray inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_next_array_advanced_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      const int64_t* fromarray,
      const int64_t* fromadvanced,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromarray inparam role: ListArray-at-array
  /// @param fromadvanced inparam role: ListArray-diff
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lenarray inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_next_array_advanced_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      const int64_t* fromarray,
      const int64_t* fromadvanced,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromarray inparam role: ListArray-at-array
  /// @param fromadvanced inparam role: ListArray-diff
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lenarray inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_next_array_advanced_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      const int64_t* fromarray,
      const int64_t* fromadvanced,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lenarray,
      int64_t lencontent);

  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromcarry inparam role: ListArray-at-array
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_carry_64(
      int32_t* tostarts,
      int32_t* tostops,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      const int64_t* fromcarry,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lencarry);
  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromcarry inparam role: ListArray-at-array
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_carry_64(
      uint32_t* tostarts,
      uint32_t* tostops,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      const int64_t* fromcarry,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lencarry);
  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param fromcarry inparam role: ListArray-at-array
  /// @param startsoffset inparam role: ListArray-starts-offset
  /// @param stopsoffset inparam role: ListArray-stops-offset
  /// @param lenstarts inparam
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_carry_64(
      int64_t* tostarts,
      int64_t* tostops,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      const int64_t* fromcarry,
      int64_t startsoffset,
      int64_t stopsoffset,
      int64_t lenstarts,
      int64_t lencarry);

  /// @param tocarry outparam
  /// @param at inparam role: ListArray-at
  /// @param length inparam
  /// @param size inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_next_at_64(
      int64_t* tocarry,
      int64_t at,
      int64_t length,
      int64_t size);

  /// @param tocarry outparam
  /// @param regular_start inparam
  /// @param step inparam
  /// @param length inparam
  /// @param size inparam
  /// @param nextsize inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_next_range_64(
      int64_t* tocarry,
      int64_t regular_start,
      int64_t step,
      int64_t length,
      int64_t size,
      int64_t nextsize);

  /// @param toadvanced outparam
  /// @param fromadvanced inparam role: RegularArray-diff
  /// @param length inparam
  /// @param nextsize inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_next_range_spreadadvanced_64(
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      int64_t length,
      int64_t nextsize);

  /// @param toarray outparam
  /// @param fromarray inparam role: ListArray-at-array
  /// @param lenarray inparam
  /// @param size inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_next_array_regularize_64(
      int64_t* toarray,
      const int64_t* fromarray,
      int64_t lenarray,
      int64_t size);

  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromarray inparam role: ListArray-at-array
  /// @param length inparam
  /// @param lenarray inparam
  /// @param size inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_next_array_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const int64_t* fromarray,
      int64_t length,
      int64_t lenarray,
      int64_t size);

  /// @param tocarry outparam
  /// @param toadvanced outparam
  /// @param fromadvanced inparam role: RegularArray-diff
  /// @param fromarray inparam role: ListArray-at-array
  /// @param length inparam
  /// @param lenarray inparam
  /// @param size inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_next_array_advanced_64(
      int64_t* tocarry,
      int64_t* toadvanced,
      const int64_t* fromadvanced,
      const int64_t* fromarray,
      int64_t length,
      int64_t lenarray,
      int64_t size);

  /// @param tocarry outparam
  /// @param fromcarry inparam role: RegularArray-diff
  /// @param lencarry inparam
  /// @param size inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_carry_64(
      int64_t* tocarry,
      const int64_t* fromcarry,
      int64_t lencarry,
      int64_t size);

  /// @param numnull outparam role: pointer
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_numnull(
      int64_t* numnull,
      const int32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex);
  /// @param numnull outparam role: pointer
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_numnull(
      int64_t* numnull,
      const uint32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex);
  /// @param numnull outparam role: pointer
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_numnull(
      int64_t* numnull,
      const int64_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex);

  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_getitem_nextcarry_outindex_64(
      int64_t* tocarry,
      int32_t* toindex,
      const int32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_getitem_nextcarry_outindex_64(
      int64_t* tocarry,
      uint32_t* toindex,
      const uint32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_getitem_nextcarry_outindex_64(
      int64_t* tocarry,
      int64_t* toindex,
      const int64_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);

  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64(
      int64_t* tocarry,
      int64_t* toindex,
      const int32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64(
      int64_t* tocarry,
      int64_t* toindex,
      const uint32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64(
      int64_t* tocarry,
      int64_t* toindex,
      const int64_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);

  /// @param tooffsets outparam
  /// @param tononzero outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets instance: 0
  /// @param offsetsoffset inparam role: ListOffsetArray-offsets-offset instance: 0
  /// @param length inparam
  /// @param nonzero inparam role: ListOffsetArray-offsets instance: 1
  /// @param nonzerooffset inparam role: ListOffsetArray-offsets-offset instance: 1
  /// @param nonzerolength inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_getitem_adjust_offsets_64(
      int64_t* tooffsets,
      int64_t* tononzero,
      const int64_t* fromoffsets,
      int64_t offsetsoffset,
      int64_t length,
      const int64_t* nonzero,
      int64_t nonzerooffset,
      int64_t nonzerolength);

  /// @param tooffsets outparam
  /// @param tononzero outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets instance: 0
  /// @param offsetsoffset inparam role: ListOffsetArray-offsets-offset instance: 0
  /// @param length inparam
  /// @param index inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param indexlength inparam
  /// @param nonzero inparam role: ListOffsetArray-offsets instance: 1
  /// @param nonzerooffset inparam role: ListOffsetArray-offsets-offset instance: 1
  /// @param nonzerolength inparam
  /// @param originalmask inparam role: IndexedArray-mask
  /// @param maskoffset inparam role: IndexedArray-mask-offset
  /// @param masklength inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray_getitem_adjust_offsets_index_64(
      int64_t* tooffsets,
      int64_t* tononzero,
      const int64_t* fromoffsets,
      int64_t offsetsoffset,
      int64_t length,
      const int64_t* index,
      int64_t indexoffset,
      int64_t indexlength,
      const int64_t* nonzero,
      int64_t nonzerooffset,
      int64_t nonzerolength,
      const int8_t* originalmask,
      int64_t maskoffset,
      int64_t masklength);

  /// @param tomask outparam
  /// @param toindex outparam
  /// @param tononzero outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 0
  /// @param fromindexoffset inparam role: IndexedArray-index-offset instance: 0
  /// @param fromindexlength inparam
  /// @param nonzero inparam role: IndexedArray-index instance: 1
  /// @param nonzerooffset inparam role: IndexedArray-index-offset instance: 1
  /// @param nonzerolength inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_getitem_adjust_outindex_64(
      int8_t* tomask,
      int64_t* toindex,
      int64_t* tononzero,
      const int64_t* fromindex,
      int64_t fromindexoffset,
      int64_t fromindexlength,
      const int64_t* nonzero,
      int64_t nonzerooffset,
      int64_t nonzerolength);

  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_getitem_nextcarry_64(
      int64_t* tocarry,
      const int32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_getitem_nextcarry_64(
      int64_t* tocarry,
      const uint32_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_getitem_nextcarry_64(
      int64_t* tocarry,
      const int64_t* fromindex,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencontent);

  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 0
  /// @param fromcarry inparam role: ListArray-at-array
  /// @param indexoffset inparam role: IndexedArray-index-offset instance: 0
  /// @param lenindex inparam
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_getitem_carry_64(
      int32_t* toindex,
      const int32_t* fromindex,
      const int64_t* fromcarry,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencarry);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 0
  /// @param fromcarry inparam role: ListArray-at-array
  /// @param indexoffset inparam role: IndexedArray-index-offset instance: 0
  /// @param lenindex inparam
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_getitem_carry_64(
      uint32_t* toindex,
      const uint32_t* fromindex,
      const int64_t* fromcarry,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencarry);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index instance: 0
  /// @param fromcarry inparam role: ListArray-at-array
  /// @param indexoffset inparam role: IndexedArray-index-offset instance: 0
  /// @param lenindex inparam
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_getitem_carry_64(
      int64_t* toindex,
      const int64_t* fromindex,
      const int64_t* fromcarry,
      int64_t indexoffset,
      int64_t lenindex,
      int64_t lencarry);

  /// @param size outparam role: pointer
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_regular_index_getsize(
      int64_t* size,
      const int8_t* fromtags,
      int64_t tagsoffset,
      int64_t length);

  /// @param toindex outparam
  /// @param current outparam
  /// @param size inparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_regular_index(
      int32_t* toindex,
      int32_t* current,
      int64_t size,
      const int8_t* fromtags,
      int64_t tagsoffset,
      int64_t length);
  /// @param toindex outparam
  /// @param current outparam
  /// @param size inparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_regular_index(
      uint32_t* toindex,
      uint32_t* current,
      int64_t size,
      const int8_t* fromtags,
      int64_t tagsoffset,
      int64_t length);
  /// @param toindex outparam
  /// @param current outparam
  /// @param size inparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_regular_index(
      int64_t* toindex,
      int64_t* current,
      int64_t size,
      const int8_t* fromtags,
      int64_t tagsoffset,
      int64_t length);

  /// @param lenout outparam role: pointer
  /// @param tocarry outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param length inparam
  /// @param which inparam role: UnionArray-innerwhich
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_project_64(
      int64_t* lenout,
      int64_t* tocarry,
      const int8_t* fromtags,
      int64_t tagsoffset,
      const int32_t* fromindex,
      int64_t indexoffset,
      int64_t length,
      int64_t which);
  /// @param lenout outparam role: pointer
  /// @param tocarry outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param length inparam
  /// @param which inparam role: UnionArray-innerwhich
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_project_64(
      int64_t* lenout,
      int64_t* tocarry,
      const int8_t* fromtags,
      int64_t tagsoffset,
      const uint32_t* fromindex,
      int64_t indexoffset,
      int64_t length,
      int64_t which);
  /// @param lenout outparam role: pointer
  /// @param tocarry outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param tagsoffset inparam role: UnionArray-tags-offset
  /// @param fromindex inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param length inparam
  /// @param which inparam role: UnionArray-innerwhich
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_project_64(
      int64_t* lenout,
      int64_t* tocarry,
      const int8_t* fromtags,
      int64_t tagsoffset,
      const int64_t* fromindex,
      int64_t indexoffset,
      int64_t length,
      int64_t which);

  /// @param outindex outparam
  /// @param index inparam role: IndexedArray-index
  /// @param indexoffset inparam role: IndexedArray-index-offset
  /// @param indexlength inparam
  /// @param repetitions inparam
  /// @param regularsize inparam
  EXPORT_SYMBOL struct Error
    awkward_missing_repeat_64(
      int64_t* outindex,
      const int64_t* index,
      int64_t indexoffset,
      int64_t indexlength,
      int64_t repetitions,
      int64_t regularsize);

  /// @param multistarts outparam
  /// @param multistops outparam
  /// @param singleoffsets inparam role: RegularArray-diff
  /// @param regularsize inparam role: RegularArray-size
  /// @param regularlength inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_getitem_jagged_expand_64(
      int64_t* multistarts,
      int64_t* multistops,
      const int64_t* singleoffsets,
      int64_t regularsize,
      int64_t regularlength);

  /// @param multistarts outparam
  /// @param multistops outparam
  /// @param singleoffsets inparam role: ListArray-diff
  /// @param tocarry outparam
  /// @param fromstarts inparam role: ListArray-starts-samediff
  /// @param fromstartsoffset inparam role: ListArray-starts-offset
  /// @param fromstops inparam role: ListArray-stops-samediff
  /// @param fromstopsoffset inparam role: ListArray-stops-offset
  /// @param jaggedsize inparam role: ListArray-jaggedsize
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_jagged_expand_64(
      int64_t* multistarts,
      int64_t* multistops,
      const int64_t* singleoffsets,
      int64_t* tocarry,
      const int32_t* fromstarts,
      int64_t fromstartsoffset,
      const int32_t* fromstops,
      int64_t fromstopsoffset,
      int64_t jaggedsize,
      int64_t length);
  /// @param multistarts outparam
  /// @param multistops outparam
  /// @param singleoffsets inparam role: ListArray-diff
  /// @param tocarry outparam
  /// @param fromstarts inparam role: ListArray-starts-samediff
  /// @param fromstartsoffset inparam role: ListArray-starts-offset
  /// @param fromstops inparam role: ListArray-stops-samediff
  /// @param fromstopsoffset inparam role: ListArray-stops-offset
  /// @param jaggedsize inparam role: ListArray-jaggedsize
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_jagged_expand_64(
      int64_t* multistarts,
      int64_t* multistops,
      const int64_t* singleoffsets,
      int64_t* tocarry,
      const uint32_t* fromstarts,
      int64_t fromstartsoffset,
      const uint32_t* fromstops,
      int64_t fromstopsoffset,
      int64_t jaggedsize,
      int64_t length);
  /// @param multistarts outparam
  /// @param multistops outparam
  /// @param singleoffsets inparam role: ListArray-diff
  /// @param tocarry outparam
  /// @param fromstarts inparam role: ListArray-starts-samediff
  /// @param fromstartsoffset inparam role: ListArray-starts-offset
  /// @param fromstops inparam role: ListArray-stops-samediff
  /// @param fromstopsoffset inparam role: ListArray-stops-offset
  /// @param jaggedsize inparam role: ListArray-jaggedsize
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_jagged_expand_64(
      int64_t* multistarts,
      int64_t* multistops,
      const int64_t* singleoffsets,
      int64_t* tocarry,
      const int64_t* fromstarts,
      int64_t fromstartsoffset,
      const int64_t* fromstops,
      int64_t fromstopsoffset,
      int64_t jaggedsize,
      int64_t length);

  /// @param carrylen outparam role: pointer
  /// @param slicestarts inparam role: ListArray-starts
  /// @param slicestartsoffset inparam role: ListArray-starts-offset
  /// @param slicestops inparam role: ListArray-stops
  /// @param slicestopsoffset inparam role: ListArray-stops-offset
  /// @param sliceouterlen inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray_getitem_jagged_carrylen_64(
      int64_t* carrylen,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen);

  /// @param tooffsets outparam
  /// @param tocarry outparam
  /// @param slicestarts inparam role: ListArray-starts
  /// @param slicestartsoffset inparam role: ListArray-starts-offset
  /// @param slicestops inparam role: ListArray-stops
  /// @param slicestopsoffset inparam role: ListArray-stops-offset
  /// @param sliceouterlen inparam
  /// @param sliceindex inparam role: ListArray-at-array
  /// @param sliceindexoffset inparam role: ListArray-diff-offset
  /// @param sliceinnerlen inparam
  /// @param fromstarts inparam role: ListArray-starts-samediff
  /// @param fromstartsoffset inparam role: ListArray-starts-offset
  /// @param fromstops inparam role: ListArray-stops-samediff
  /// @param fromstopsoffset inparam role: ListArray-stops-offset
  /// @param contentlen inparam role: ListArray-contentlen
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_jagged_apply_64(
      int64_t* tooffsets,
      int64_t* tocarry,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen,
      const int64_t* sliceindex,
      int64_t sliceindexoffset,
      int64_t sliceinnerlen,
      const int32_t* fromstarts,
      int64_t fromstartsoffset,
      const int32_t* fromstops,
      int64_t fromstopsoffset,
      int64_t contentlen);
  /// @param tooffsets outparam
  /// @param tocarry outparam
  /// @param slicestarts inparam role: ListArray-starts
  /// @param slicestartsoffset inparam role: ListArray-starts-offset
  /// @param slicestops inparam role: ListArray-stops
  /// @param slicestopsoffset inparam role: ListArray-stops-offset
  /// @param sliceouterlen inparam
  /// @param sliceindex inparam role: ListArray-at-array
  /// @param sliceindexoffset inparam role: ListArray-diff-offset
  /// @param sliceinnerlen inparam
  /// @param fromstarts inparam role: ListArray-starts-samediff
  /// @param fromstartsoffset inparam role: ListArray-starts-offset
  /// @param fromstops inparam role: ListArray-stops-samediff
  /// @param fromstopsoffset inparam role: ListArray-stops-offset
  /// @param contentlen inparam role: ListArray-contentlen
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_jagged_apply_64(
      int64_t* tooffsets,
      int64_t* tocarry,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen,
      const int64_t* sliceindex,
      int64_t sliceindexoffset,
      int64_t sliceinnerlen,
      const uint32_t* fromstarts,
      int64_t fromstartsoffset,
      const uint32_t* fromstops,
      int64_t fromstopsoffset,
      int64_t contentlen);
  /// @param tooffsets outparam
  /// @param tocarry outparam
  /// @param slicestarts inparam role: ListArray-starts
  /// @param slicestartsoffset inparam role: ListArray-starts-offset
  /// @param slicestops inparam role: ListArray-stops
  /// @param slicestopsoffset inparam role: ListArray-stops-offset
  /// @param sliceouterlen inparam
  /// @param sliceindex inparam role: ListArray-at-array
  /// @param sliceindexoffset inparam role: ListArray-diff-offset
  /// @param sliceinnerlen inparam
  /// @param fromstarts inparam role: ListArray-starts-samediff
  /// @param fromstartsoffset inparam role: ListArray-starts-offset
  /// @param fromstops inparam role: ListArray-stops-samediff
  /// @param fromstopsoffset inparam role: ListArray-stops-offset
  /// @param contentlen inparam role: ListArray-contentlen
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_jagged_apply_64(
      int64_t* tooffsets,
      int64_t* tocarry,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen,
      const int64_t* sliceindex,
      int64_t sliceindexoffset,
      int64_t sliceinnerlen,
      const int64_t* fromstarts,
      int64_t fromstartsoffset,
      const int64_t* fromstops,
      int64_t fromstopsoffset,
      int64_t contentlen);

  /// @param numvalid outparam role: pointer
  /// @param slicestarts inparam role: ListArray-starts
  /// @param slicestartsoffset inparam role: ListArray-starts-offset
  /// @param slicestops inparam role: ListArray-stops
  /// @param slicestopsoffset inparam role: ListArray-stops-offset
  /// @param length inparam
  /// @param missing inparam role: ListArray-diff
  /// @param missingoffset inparam role: ListArray-diff-offset
  /// @param missinglength inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray_getitem_jagged_numvalid_64(
      int64_t* numvalid,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t length,
      const int64_t* missing,
      int64_t missingoffset,
      int64_t missinglength);

  /// @param tocarry outparam
  /// @param tosmalloffsets outparam
  /// @param tolargeoffsets outparam
  /// @param slicestarts inparam role: ListArray-starts
  /// @param slicestartsoffset inparam role: ListArray-starts-offset
  /// @param slicestops inparam role: ListArray-stops
  /// @param slicestopsoffset inparam role: ListArray-stops-offset
  /// @param length inparam
  /// @param missing inparam role: ListArray-diff
  /// @param missingoffset inparam role: ListArray-diff-offset
  EXPORT_SYMBOL struct Error
    awkward_ListArray_getitem_jagged_shrink_64(
      int64_t* tocarry,
      int64_t* tosmalloffsets,
      int64_t* tolargeoffsets,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t length,
      const int64_t* missing,
      int64_t missingoffset);

  /// @param tooffsets outparam
  /// @param slicestarts inparam role: ListArray-starts-samediff instance: 0
  /// @param slicestartsoffset inparam role: ListArray-starts-offset instance: 0
  /// @param slicestops inparam role: ListArray-stops-samediff instance: 0
  /// @param slicestopsoffset inparam role: ListArray-stops-offset instance: 0
  /// @param sliceouterlen inparam
  /// @param fromstarts inparam role: ListArray-starts-samediff instance: 1
  /// @param fromstartsoffset inparam role: ListArray-starts-offset instance: 1
  /// @param fromstops inparam role: ListArray-stops-samediff instance: 1
  /// @param fromstopsoffset inparam role: ListArray-stops-offset instance: 1
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_getitem_jagged_descend_64(
      int64_t* tooffsets,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen,
      const int32_t* fromstarts,
      int64_t fromstartsoffset,
      const int32_t* fromstops,
      int64_t fromstopsoffset);
  /// @param tooffsets outparam
  /// @param slicestarts inparam role: ListArray-starts-samediff instance: 0
  /// @param slicestartsoffset inparam role: ListArray-starts-offset instance: 0
  /// @param slicestops inparam role: ListArray-stops-samediff instance: 0
  /// @param slicestopsoffset inparam role: ListArray-stops-offset instance: 0
  /// @param sliceouterlen inparam
  /// @param fromstarts inparam role: ListArray-starts-samediff instance: 1
  /// @param fromstartsoffset inparam role: ListArray-starts-offset instance: 1
  /// @param fromstops inparam role: ListArray-stops-samediff instance: 1
  /// @param fromstopsoffset inparam role: ListArray-stops-offset instance: 1
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_getitem_jagged_descend_64(
      int64_t* tooffsets,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen,
      const uint32_t* fromstarts,
      int64_t fromstartsoffset,
      const uint32_t* fromstops,
      int64_t fromstopsoffset);
  /// @param tooffsets outparam
  /// @param slicestarts inparam role: ListArray-starts-samediff instance: 0
  /// @param slicestartsoffset inparam role: ListArray-starts-offset instance: 0
  /// @param slicestops inparam role: ListArray-stops-samediff instance: 0
  /// @param slicestopsoffset inparam role: ListArray-stops-offset instance: 0
  /// @param sliceouterlen inparam
  /// @param fromstarts inparam role: ListArray-starts-samediff instance: 1
  /// @param fromstartsoffset inparam role: ListArray-starts-offset instance: 1
  /// @param fromstops inparam role: ListArray-stops-samediff instance: 1
  /// @param fromstopsoffset inparam role: ListArray-stops-offset instance: 1
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_getitem_jagged_descend_64(
      int64_t* tooffsets,
      const int64_t* slicestarts,
      int64_t slicestartsoffset,
      const int64_t* slicestops,
      int64_t slicestopsoffset,
      int64_t sliceouterlen,
      const int64_t* fromstarts,
      int64_t fromstartsoffset,
      const int64_t* fromstops,
      int64_t fromstopsoffset);

  EXPORT_SYMBOL int8_t
    awkward_Index8_getitem_at_nowrap(
      const int8_t* ptr,
      int64_t offset,
      int64_t at);
  EXPORT_SYMBOL uint8_t
    awkward_IndexU8_getitem_at_nowrap(
      const uint8_t* ptr,
      int64_t offset,
      int64_t at);
  EXPORT_SYMBOL int32_t
    awkward_Index32_getitem_at_nowrap(
      const int32_t* ptr,
      int64_t offset,
      int64_t at);
  EXPORT_SYMBOL uint32_t
    awkward_IndexU32_getitem_at_nowrap(
      const uint32_t* ptr,
      int64_t offset,
      int64_t at);
  EXPORT_SYMBOL int64_t
    awkward_Index64_getitem_at_nowrap(
      const int64_t* ptr,
      int64_t offset,
      int64_t at);

  EXPORT_SYMBOL bool
    awkward_NumpyArraybool_getitem_at(
      const bool* ptr,
      int64_t at);
  EXPORT_SYMBOL int8_t
    awkward_NumpyArray8_getitem_at(
      const int8_t* ptr,
      int64_t at);
  EXPORT_SYMBOL uint8_t
    awkward_NumpyArrayU8_getitem_at(
      const uint8_t* ptr,
      int64_t at);
  EXPORT_SYMBOL int16_t
    awkward_NumpyArray16_getitem_at(
      const int16_t* ptr,
      int64_t at);
  EXPORT_SYMBOL uint16_t
    awkward_NumpyArrayU16_getitem_at(
      const uint16_t* ptr,
      int64_t at);
  EXPORT_SYMBOL int32_t
    awkward_NumpyArray32_getitem_at(
      const int32_t* ptr,
      int64_t at);
  EXPORT_SYMBOL uint32_t
    awkward_NumpyArrayU32_getitem_at(
      const uint32_t* ptr,
      int64_t at);
  EXPORT_SYMBOL int64_t
    awkward_NumpyArray64_getitem_at(
      const int64_t* ptr,
      int64_t at);
  EXPORT_SYMBOL uint64_t
    awkward_NumpyArrayU64_getitem_at(
      const uint64_t* ptr,
      int64_t at);
  EXPORT_SYMBOL float
    awkward_NumpyArrayfloat32_getitem_at(
      const float* ptr,
      int64_t at);
  EXPORT_SYMBOL double
    awkward_NumpyArrayfloat64_getitem_at(
      const double* ptr,
      int64_t at);

  EXPORT_SYMBOL void
    awkward_Index8_setitem_at_nowrap(
      int8_t* ptr,
      int64_t offset,
      int64_t at,
      int8_t value);
  EXPORT_SYMBOL void
    awkward_IndexU8_setitem_at_nowrap(
      uint8_t* ptr,
      int64_t offset,
      int64_t at,
      uint8_t value);
  EXPORT_SYMBOL void
    awkward_Index32_setitem_at_nowrap(
      int32_t* ptr,
      int64_t offset,
      int64_t at,
      int32_t value);
  EXPORT_SYMBOL void
    awkward_IndexU32_setitem_at_nowrap(
      uint32_t* ptr,
      int64_t offset,
      int64_t at,
      uint32_t value);
  EXPORT_SYMBOL void
    awkward_Index64_setitem_at_nowrap(
      int64_t* ptr,
      int64_t offset,
      int64_t at,
      int64_t value);

  /// @param tomask outparam
  /// @param frommask inparam role: ByteMaskedArray-mask
  /// @param frommaskoffset inparam role: ByteMaskedArray-mask-offset
  /// @param lenmask inparam
  /// @param fromcarry inparam role: IndexedArray-index
  /// @param lencarry inparam
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_getitem_carry_64(
      int8_t* tomask,
      const int8_t* frommask,
      int64_t frommaskoffset,
      int64_t lenmask,
      const int64_t* fromcarry,
      int64_t lencarry);

  /// @param numnull outparam role: pointer
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param maskoffset inparam role: ByteMaskedArray-mask-offset
  /// @param length inparam
  /// @param validwhen inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_numnull(
      int64_t* numnull,
      const int8_t* mask,
      int64_t maskoffset,
      int64_t length,
      bool validwhen);
  /// @param tocarry outparam
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param maskoffset inparam role: ByteMaskedArray-mask-offset
  /// @param length inparam
  /// @param validwhen inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_getitem_nextcarry_64(
      int64_t* tocarry,
      const int8_t* mask,
      int64_t maskoffset,
      int64_t length,
      bool validwhen);
  /// @param tocarry outparam
  /// @param outindex outparam
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param maskoffset inparam role: ByteMaskedArray-mask-offset
  /// @param length inparam
  /// @param validwhen inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
      int64_t* tocarry,
      int64_t* outindex,
      const int8_t* mask,
      int64_t maskoffset,
      int64_t length,
      bool validwhen);

  /// @param toindex outparam
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param maskoffset inparam role: ByteMaskedArray-mask-offset
  /// @param length inparam
  /// @param validwhen inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_toIndexedOptionArray64(
      int64_t* toindex,
      const int8_t* mask,
      int64_t maskoffset,
      int64_t length,
      bool validwhen);

  /// @param index_in inparam role: IndexedArray-index instance: 0
  /// @param index_in_offset inparam role: IndexedArray-index-offset instance: 0
  /// @param offsets_in inparam role: IndexedArray-index instance: 1
  /// @param offsets_in_offset inparam role: IndexedArray-index-offset instance: 1
  /// @param mask_out outparam
  /// @param starts_out outparam
  /// @param stops_out outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
  awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
      int64_t* index_in, int64_t index_in_offset, int64_t* offsets_in,
      int64_t offsets_in_offset, int64_t* mask_out, int64_t* starts_out,
      int64_t* stops_out, int64_t length);

  /// @param index inparam role: IndexedArray-index
  /// @param index_offset inparam role: IndexedArray-index-offset
  /// @param starts_in inparam role: ListArray-starts
  /// @param starts_offset inparam role: ListArray-starts-offset
  /// @param stops_in inparam role: ListArray-stops
  /// @param stops_offset inparam role: ListArray-stops-offset
  /// @param starts_out outparam
  /// @param stops_out outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error awkward_MaskedArray32_getitem_next_jagged_project(
      int32_t* index, int64_t index_offset, int64_t* starts_in,
      int64_t starts_offset, int64_t* stops_in, int64_t stops_offset,
      int64_t* starts_out, int64_t* stops_out, int64_t length);
  /// @param index inparam role: IndexedArray-index
  /// @param index_offset inparam role: IndexedArray-index-offset
  /// @param starts_in inparam role: ListArray-starts
  /// @param starts_offset inparam role: ListArray-starts-offset
  /// @param stops_in inparam role: ListArray-stops
  /// @param stops_offset inparam role: ListArray-stops-offset
  /// @param starts_out outparam
  /// @param stops_out outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error awkward_MaskedArrayU32_getitem_next_jagged_project(
      uint32_t* index, int64_t index_offset, int64_t* starts_in,
      int64_t starts_offset, int64_t* stops_in, int64_t stops_offset,
      int64_t* starts_out, int64_t* stops_out, int64_t length);
  /// @param index inparam role: IndexedArray-index
  /// @param index_offset inparam role: IndexedArray-index-offset
  /// @param starts_in inparam role: ListArray-starts
  /// @param starts_offset inparam role: ListArray-starts-offset
  /// @param stops_in inparam role: ListArray-stops
  /// @param stops_offset inparam role: ListArray-stops-offset
  /// @param starts_out outparam
  /// @param stops_out outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error awkward_MaskedArray64_getitem_next_jagged_project(
      int64_t* index, int64_t index_offset, int64_t* starts_in,
      int64_t starts_offset, int64_t* stops_in, int64_t stops_offset,
      int64_t* starts_out, int64_t* stops_out, int64_t length);
}

#endif // AWKWARDCPU_GETITEM_H_
