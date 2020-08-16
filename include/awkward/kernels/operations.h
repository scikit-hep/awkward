// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_KERNELS_OPERATIONS_H_
#define AWKWARD_KERNELS_OPERATIONS_H_

#include "awkward/common.h"

extern "C" {
  /// @param tonum outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_num_64(
      int64_t* tonum,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t length);
  /// @param tonum outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_num_64(
      int64_t* tonum,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t length);
  /// @param tonum outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_num_64(
      int64_t* tonum,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t length);

  /// @param tonum outparam
  /// @param size inparam role: RegularArray-size
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_num_64(
      int64_t* tonum,
      int64_t size,
      int64_t length);

  /// @param tooffsets outparam
  /// @param outeroffsets inparam role: ListOffsetArray-offsets
  /// @param outeroffsetslen inparam
  /// @param inneroffsets inparam role: ListOffsetArray2-offsets
  /// @param inneroffsetslen inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray32_flatten_offsets_64(
      int64_t* tooffsets,
      const int32_t* outeroffsets,
      int64_t outeroffsetslen,
      const int64_t* inneroffsets,
      int64_t inneroffsetslen);
  /// @param tooffsets outparam
  /// @param outeroffsets inparam role: ListOffsetArray-offsets
  /// @param outeroffsetslen inparam
  /// @param inneroffsets inparam role: ListOffsetArray2-offsets
  /// @param inneroffsetslen inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArrayU32_flatten_offsets_64(
      int64_t* tooffsets,
      const uint32_t* outeroffsets,
      int64_t outeroffsetslen,
      const int64_t* inneroffsets,
      int64_t inneroffsetslen);
  /// @param tooffsets outparam
  /// @param outeroffsets inparam role: ListOffsetArray-offsets
  /// @param outeroffsetslen inparam
  /// @param inneroffsets inparam role: ListOffsetArray2-offsets
  /// @param inneroffsetslen inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray64_flatten_offsets_64(
      int64_t* tooffsets,
      const int64_t* outeroffsets,
      int64_t outeroffsetslen,
      const int64_t* inneroffsets,
      int64_t inneroffsetslen);

  /// @param outoffsets outparam
  /// @param outindex inparam role: IndexedArray-index
  /// @param outindexlength inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_flatten_none2empty_64(
      int64_t* outoffsets,
      const int32_t* outindex,
      int64_t outindexlength,
      const int64_t* offsets,
      int64_t offsetslength);
  /// @param outoffsets outparam
  /// @param outindex inparam role: IndexedArray-index
  /// @param outindexlength inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_flatten_none2empty_64(
      int64_t* outoffsets,
      const uint32_t* outindex,
      int64_t outindexlength,
      const int64_t* offsets,
      int64_t offsetslength);
  /// @param outoffsets outparam
  /// @param outindex inparam role: IndexedArray-index
  /// @param outindexlength inparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_flatten_none2empty_64(
      int64_t* outoffsets,
      const int64_t* outindex,
      int64_t outindexlength,
      const int64_t* offsets,
      int64_t offsetslength);

  /// @param total_length outparam role: pointer
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsetsraws inparam role: ListOffsetArray-rawoffsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray32_flatten_length_64(
      int64_t* total_length,
      const int8_t* fromtags,
      const int32_t* fromindex,
      int64_t length,
      int64_t** offsetsraws);
  /// @param total_length outparam role: pointer
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsetsraws inparam role: ListOffsetArray-rawoffsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArrayU32_flatten_length_64(
      int64_t* total_length,
      const int8_t* fromtags,
      const uint32_t* fromindex,
      int64_t length,
      int64_t** offsetsraws);
  /// @param total_length outparam role: pointer
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsetsraws inparam role: ListOffsetArray-rawoffsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray64_flatten_length_64(
      int64_t* total_length,
      const int8_t* fromtags,
      const int64_t* fromindex,
      int64_t length,
      int64_t** offsetsraws);

  /// @param totags outparam
  /// @param toindex outparam
  /// @param tooffsets outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param offsetsraws inparam role: ListOffsetArray-rawoffsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray32_flatten_combine_64(
      int8_t* totags,
      int64_t* toindex,
      int64_t* tooffsets,
      const int8_t* fromtags,
      const int32_t* fromindex,
      int64_t length,
      int64_t** offsetsraws);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param tooffsets outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: UnionArray-index
  /// @param length inparam
  /// @param offsetsraws inparam role: ListOffsetArray-rawoffsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArrayU32_flatten_combine_64(
      int8_t* totags,
      int64_t* toindex,
      int64_t* tooffsets,
      const int8_t* fromtags,
      const uint32_t* fromindex,
      int64_t length,
      int64_t** offsetsraws);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param tooffsets outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: UnionArray-index
  /// @param length inparam
  /// @param offsetsraws inparam role: ListOffsetArray-rawoffsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray64_flatten_combine_64(
      int8_t* totags,
      int64_t* toindex,
      int64_t* tooffsets,
      const int8_t* fromtags,
      const int64_t* fromindex,
      int64_t length,
      int64_t** offsetsraws);

  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_flatten_nextcarry_64(
      int64_t* tocarry,
      const int32_t* fromindex,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_flatten_nextcarry_64(
      int64_t* tocarry,
      const uint32_t* fromindex,
      int64_t lenindex,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param lenindex inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_flatten_nextcarry_64(
      int64_t* tocarry,
      const int64_t* fromindex,
      int64_t lenindex,
      int64_t lencontent);

  /// @param toindex outparam
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_overlay_mask8_to64(
      int64_t* toindex,
      const int8_t* mask,
      const int32_t* fromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_overlay_mask8_to64(
      int64_t* toindex,
      const int8_t* mask,
      const uint32_t* fromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param mask inparam role: ByteMaskedArray-mask
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_overlay_mask8_to64(
      int64_t* toindex,
      const int8_t* mask,
      const int64_t* fromindex,
      int64_t length);

  /// @param tomask outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_mask8(
      int8_t* tomask,
      const int32_t* fromindex,
      int64_t length);
  /// @param tomask outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_mask8(
      int8_t* tomask,
      const uint32_t* fromindex,
      int64_t length);
  /// @param tomask outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_mask8(
      int8_t* tomask,
      const int64_t* fromindex,
      int64_t length);

  /// @param tomask outparam
  /// @param frommask inparam role: ByteMaskedArray-mask
  /// @param length inparam
  /// @param validwhen inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_mask8(
      int8_t* tomask,
      const int8_t* frommask,
      int64_t length,
      bool validwhen);

  /// @param tomask outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_zero_mask8(
      int8_t* tomask,
      int64_t length);

  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_simplify32_to64(
      int64_t* toindex,
      const int32_t* outerindex,
      int64_t outerlength,
      const int32_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_simplifyU32_to64(
      int64_t* toindex,
      const int32_t* outerindex,
      int64_t outerlength,
      const uint32_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_simplify64_to64(
      int64_t* toindex,
      const int32_t* outerindex,
      int64_t outerlength,
      const int64_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_simplify32_to64(
      int64_t* toindex,
      const uint32_t* outerindex,
      int64_t outerlength,
      const int32_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_simplifyU32_to64(
      int64_t* toindex,
      const uint32_t* outerindex,
      int64_t outerlength,
      const uint32_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_simplify64_to64(
      int64_t* toindex,
      const uint32_t* outerindex,
      int64_t outerlength,
      const int64_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_simplify32_to64(
      int64_t* toindex,
      const int64_t* outerindex,
      int64_t outerlength,
      const int32_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_simplifyU32_to64(
      int64_t* toindex,
      const int64_t* outerindex,
      int64_t outerlength,
      const uint32_t* innerindex,
      int64_t innerlength);
  /// @param toindex outparam
  /// @param outerindex inparam role: IndexedArray-index
  /// @param outerlength inparam
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param innerlength inparam role: IndexedArray-length
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_simplify64_to64(
      int64_t* toindex,
      const int64_t* outerindex,
      int64_t outerlength,
      const int64_t* innerindex,
      int64_t innerlength);

  /// @param tooffsets outparam
  /// @param length inparam
  /// @param size inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_compact_offsets64(
      int64_t* tooffsets,
      int64_t length,
      int64_t size);

  /// @param tooffsets outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_compact_offsets_64(
      int64_t* tooffsets,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t length);
  /// @param tooffsets outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_compact_offsets_64(
      int64_t* tooffsets,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t length);
  /// @param tooffsets outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_compact_offsets_64(
      int64_t* tooffsets,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t length);

  /// @param tooffsets outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray32_compact_offsets_64(
      int64_t* tooffsets,
      const int32_t* fromoffsets,
      int64_t length);
  /// @param tooffsets outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArrayU32_compact_offsets_64(
      int64_t* tooffsets,
      const uint32_t* fromoffsets,
      int64_t length);
  /// @param tooffsets outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray64_compact_offsets_64(
      int64_t* tooffsets,
      const int64_t* fromoffsets,
      int64_t length);

  /// @param tocarry outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lencontent inparam role: ListArray-length
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_broadcast_tooffsets_64(
      int64_t* tocarry,
      const int64_t* fromoffsets,
      int64_t offsetslength,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lencontent inparam role: ListArray-length
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_broadcast_tooffsets_64(
      int64_t* tocarry,
      const int64_t* fromoffsets,
      int64_t offsetslength,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t lencontent);
  /// @param tocarry outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lencontent inparam role: ListArray-length
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_broadcast_tooffsets_64(
      int64_t* tocarry,
      const int64_t* fromoffsets,
      int64_t offsetslength,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t lencontent);

  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  /// @param size inparam role: RegularArray-size
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_broadcast_tooffsets_64(
      const int64_t* fromoffsets,
      int64_t offsetslength,
      int64_t size);

  /// @param tocarry outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_broadcast_tooffsets_size1_64(
      int64_t* tocarry,
      const int64_t* fromoffsets,
      int64_t offsetslength);

  /// @param size outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray32_toRegularArray(
      int64_t* size,
      const int32_t* fromoffsets,
      int64_t offsetslength);
  /// @param size outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArrayU32_toRegularArray(
      int64_t* size,
      const uint32_t* fromoffsets,
      int64_t offsetslength);
  /// @param size outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param offsetslength inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray64_toRegularArray(
      int64_t* size,
      const int64_t* fromoffsets,
      int64_t offsetslength);

  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_frombool(
      bool* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_frombool(
      int8_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_frombool(
      int16_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_frombool(
      int32_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_frombool(
      int64_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_frombool(
      uint8_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_frombool(
      uint16_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_frombool(
      uint32_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_frombool(
      uint64_t* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_frombool(
      float* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_frombool(
      double* toptr,
      int64_t tooffset,
      const bool* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromint8(
      bool* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromint8(
      int8_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromint8(
      int16_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromint8(
      int32_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromint8(
      int64_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromint8(
      uint8_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromint8(
      uint16_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromint8(
      uint32_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromint8(
      uint64_t* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromint8(
      float* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromint8(
      double* toptr,
      int64_t tooffset,
      const int8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromint16(
      bool* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromint16(
      int8_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromint16(
      int16_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromint16(
      int32_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromint16(
      int64_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromint16(
      uint8_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromint16(
      uint16_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromint16(
      uint32_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromint16(
      uint64_t* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromint16(
      float* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromint16(
      double* toptr,
      int64_t tooffset,
      const int16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromint32(
      bool* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromint32(
      int8_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromint32(
      int16_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromint32(
      int32_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromint32(
      int64_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromint32(
      uint8_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromint32(
      uint16_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromint32(
      uint32_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromint32(
      uint64_t* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromint32(
      float* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromint32(
      double* toptr,
      int64_t tooffset,
      const int32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromint64(
      bool* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromint64(
      int8_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromint64(
      int16_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromint64(
      int32_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromint64(
      int64_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromint64(
      uint8_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromint64(
      uint16_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromint64(
      uint32_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromint64(
      uint64_t* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromint64(
      float* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromint64(
      double* toptr,
      int64_t tooffset,
      const int64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromuint8(
      bool* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromuint8(
      int8_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromuint8(
      int16_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromuint8(
      int32_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromuint8(
      int64_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromuint8(
      uint8_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromuint8(
      uint16_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromuint8(
      uint32_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromuint8(
      uint64_t* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromuint8(
      float* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromuint8(
      double* toptr,
      int64_t tooffset,
      const uint8_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromuint16(
      bool* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromuint16(
      int8_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromuint16(
      int16_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromuint16(
      int32_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromuint16(
      int64_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromuint16(
      uint8_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromuint16(
      uint16_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromuint16(
      uint32_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromuint16(
      uint64_t* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromuint16(
      float* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromuint16(
      double* toptr,
      int64_t tooffset,
      const uint16_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromuint32(
      bool* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromuint32(
      int8_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromuint32(
      int16_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromuint32(
      int32_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromuint32(
      int64_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromuint32(
      uint8_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromuint32(
      uint16_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromuint32(
      uint32_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromuint32(
      uint64_t* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromuint32(
      float* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromuint32(
      double* toptr,
      int64_t tooffset,
      const uint32_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromuint64(
      bool* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromuint64(
      int8_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromuint64(
      int16_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromuint64(
      int32_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromuint64(
      int64_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromuint64(
      uint8_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromuint64(
      uint16_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromuint64(
      uint32_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromuint64(
      uint64_t* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromuint64(
      float* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromuint64(
      double* toptr,
      int64_t tooffset,
      const uint64_t* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromfloat32(
      bool* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromfloat32(
      int8_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromfloat32(
      int16_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromfloat32(
      int32_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromfloat32(
      int64_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromfloat32(
      uint8_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromfloat32(
      uint16_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromfloat32(
      uint32_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromfloat32(
      uint64_t* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromfloat32(
      float* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromfloat32(
      double* toptr,
      int64_t tooffset,
      const float* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tobool_fromfloat64(
      bool* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint8_fromfloat64(
      int8_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint16_fromfloat64(
      int16_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint32_fromfloat64(
      int32_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_toint64_fromfloat64(
      int64_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint8_fromfloat64(
      uint8_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint16_fromfloat64(
      uint16_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint32_fromfloat64(
      uint32_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_touint64_fromfloat64(
      uint64_t* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat32_fromfloat64(
      float* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);
  /// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_NumpyArray_fill_tofloat64_fromfloat64(
      double* toptr,
      int64_t tooffset,
      const double* fromptr,
      int64_t length);

  /// @param tostarts outparam
  /// @param tostartsoffset inparam
  /// @param tostops outparam
  /// @param tostopsoffset inparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray_fill_to64_from32(
      int64_t* tostarts,
      int64_t tostartsoffset,
      int64_t* tostops,
      int64_t tostopsoffset,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t length,
      int64_t base);
  /// @param tostarts outparam
  /// @param tostartsoffset inparam
  /// @param tostops outparam
  /// @param tostopsoffset inparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray_fill_to64_fromU32(
      int64_t* tostarts,
      int64_t tostartsoffset,
      int64_t* tostops,
      int64_t tostopsoffset,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t length,
      int64_t base);
  /// @param tostarts outparam
  /// @param tostartsoffset inparam
  /// @param tostops outparam
  /// @param tostopsoffset inparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray_fill_to64_from64(
      int64_t* tostarts,
      int64_t tostartsoffset,
      int64_t* tostops,
      int64_t tostopsoffset,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t length,
      int64_t base);

  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_fill_to64_from32(
      int64_t* toindex,
      int64_t toindexoffset,
      const int32_t* fromindex,
      int64_t length,
      int64_t base);
  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_fill_to64_fromU32(
      int64_t* toindex,
      int64_t toindexoffset,
      const uint32_t* fromindex,
      int64_t length,
      int64_t base);
  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_fill_to64_from64(
      int64_t* toindex,
      int64_t toindexoffset,
      const int64_t* fromindex,
      int64_t length,
      int64_t base);

  /// @param toindex outparam
  /// @param toindexoffset inparam role: IndexedArray-index-offset
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray_fill_to64_count(
      int64_t* toindex,
      int64_t toindexoffset,
      int64_t length,
      int64_t base);

  /// @param totags outparam
  /// @param totagsoffset inparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_filltags_to8_from8(
      int8_t* totags,
      int64_t totagsoffset,
      const int8_t* fromtags,
      int64_t length,
      int64_t base);

  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillindex_to64_from32(
      int64_t* toindex,
      int64_t toindexoffset,
      const int32_t* fromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillindex_to64_fromU32(
      int64_t* toindex,
      int64_t toindexoffset,
      const uint32_t* fromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillindex_to64_from64(
      int64_t* toindex,
      int64_t toindexoffset,
      const int64_t* fromindex,
      int64_t length);

  /// @param totags outparam
  /// @param totagsoffset inparam
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_filltags_to8_const(
      int8_t* totags,
      int64_t totagsoffset,
      int64_t length,
      int64_t base);

  /// @param toindex outparam
  /// @param toindexoffset inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillindex_to64_count(
      int64_t* toindex,
      int64_t toindexoffset,
      int64_t length);

  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_simplify8_32_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const int32_t* outerindex,
      const int8_t* innertags,
      const int32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_simplify8_U32_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const int32_t* outerindex,
      const int8_t* innertags,
      const uint32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_simplify8_64_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const int32_t* outerindex,
      const int8_t* innertags,
      const int64_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_simplify8_32_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const uint32_t* outerindex,
      const int8_t* innertags,
      const int32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_simplify8_U32_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const uint32_t* outerindex,
      const int8_t* innertags,
      const uint32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_simplify8_64_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const uint32_t* outerindex,
      const int8_t* innertags,
      const int64_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_simplify8_32_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const int64_t* outerindex,
      const int8_t* innertags,
      const int32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_simplify8_U32_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const int64_t* outerindex,
      const int8_t* innertags,
      const uint32_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param outertags inparam role: UnionArray-tags
  /// @param outerindex inparam role: IndexedArray-index
  /// @param innertags inparam role: UnionArray2-tags
  /// @param innerindex inparam role: IndexedArray2-index
  /// @param towhich inparam
  /// @param innerwhich inparam role: UnionArray1-which
  /// @param outerwhich inparam role: UnionArray2-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_simplify8_64_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* outertags,
      const int64_t* outerindex,
      const int8_t* innertags,
      const int64_t* innerindex,
      int64_t towhich,
      int64_t innerwhich,
      int64_t outerwhich,
      int64_t length,
      int64_t base);

  /// @param totags outparam
  /// @param toindex outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param towhich inparam
  /// @param fromwhich inparam role: UnionArray-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_simplify_one_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* fromtags,
      const int32_t* fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param towhich inparam
  /// @param fromwhich inparam role: UnionArray-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_simplify_one_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* fromtags,
      const uint32_t* fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base);
  /// @param totags outparam
  /// @param toindex outparam
  /// @param fromtags inparam role: UnionArray-tags
  /// @param fromindex inparam role: IndexedArray-index
  /// @param towhich inparam
  /// @param fromwhich inparam role: UnionArray-which
  /// @param length inparam
  /// @param base inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_simplify_one_to8_64(
      int8_t* totags,
      int64_t* toindex,
      const int8_t* fromtags,
      const int64_t* fromindex,
      int64_t towhich,
      int64_t fromwhich,
      int64_t length,
      int64_t base);

  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillna_from32_to64(
      int64_t* toindex,
      const int32_t* fromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillna_fromU32_to64(
      int64_t* toindex,
      const uint32_t* fromindex,
      int64_t length);
  /// @param toindex outparam
  /// @param fromindex inparam role: IndexedArray-index
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_UnionArray_fillna_from64_to64(
      int64_t* toindex,
      const int64_t* fromindex,
      int64_t length);

  /// @param toindex outparam
  /// @param frommask inparam role: ByteMaskedArray-mask
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
      int64_t* toindex,
      const int8_t* frommask,
      int64_t length);

  /// @param toindex outparam
  /// @param target inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_index_rpad_and_clip_axis0_64(
      int64_t* toindex,
      int64_t target,
      int64_t length);
  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param target inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_index_rpad_and_clip_axis1_64(
      int64_t* tostarts,
      int64_t* tostops,
      int64_t target,
      int64_t length);

  /// @param toindex outparam
  /// @param target inparam
  /// @param size inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_rpad_and_clip_axis1_64(
      int64_t* toindex,
      int64_t target,
      int64_t size,
      int64_t length);

  /// @param tomin outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_min_range(
      int64_t* tomin,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t lenstarts);
  /// @param tomin outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_min_range(
      int64_t* tomin,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t lenstarts);
  /// @param tomin outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_min_range(
      int64_t* tomin,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t lenstarts);

  /// @param tomin outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param target inparam
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_rpad_and_clip_length_axis1(
      int64_t* tomin,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int64_t target,
      int64_t lenstarts);
  /// @param tomin outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param target inparam
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_rpad_and_clip_length_axis1(
      int64_t* tomin,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      int64_t target,
      int64_t lenstarts);
  /// @param tomin outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param target inparam
  /// @param lenstarts inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_rpad_and_clip_length_axis1(
      int64_t* tomin,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t target,
      int64_t lenstarts);

  /// @param toindex outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param target inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_rpad_axis1_64(
      int64_t* toindex,
      const int32_t* fromstarts,
      const int32_t* fromstops,
      int32_t* tostarts,
      int32_t* tostops,
      int64_t target,
      int64_t length);
  /// @param toindex outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param target inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_rpad_axis1_64(
      int64_t* toindex,
      const uint32_t* fromstarts,
      const uint32_t* fromstops,
      uint32_t* tostarts,
      uint32_t* tostops,
      int64_t target,
      int64_t length);
  /// @param toindex outparam
  /// @param fromstarts inparam role: ListArray-starts
  /// @param fromstops inparam role: ListArray-stops
  /// @param tostarts outparam
  /// @param tostops outparam
  /// @param target inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_rpad_axis1_64(
      int64_t* toindex,
      const int64_t* fromstarts,
      const int64_t* fromstops,
      int64_t* tostarts,
      int64_t* tostops,
      int64_t target,
      int64_t length);

  /// @param toindex outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  /// @param target inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
      int64_t* toindex,
      const int32_t* fromoffsets,
      int64_t length,
      int64_t target);
  /// @param toindex outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  /// @param target inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
      int64_t* toindex,
      const uint32_t* fromoffsets,
      int64_t length,
      int64_t target);
  /// @param toindex outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  /// @param target inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
      int64_t* toindex,
      const int64_t* fromoffsets,
      int64_t length,
      int64_t target);

  /// @param tooffsets outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param fromlength inparam
  /// @param target inparam
  /// @param tolength outparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray32_rpad_length_axis1(
      int32_t* tooffsets,
      const int32_t* fromoffsets,
      int64_t fromlength,
      int64_t target,
      int64_t* tolength);
  /// @param tooffsets outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param fromlength inparam
  /// @param target inparam
  /// @param tolength outparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArrayU32_rpad_length_axis1(
      uint32_t* tooffsets,
      const uint32_t* fromoffsets,
      int64_t fromlength,
      int64_t target,
      int64_t* tolength);
  /// @param tooffsets outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param fromlength inparam
  /// @param target inparam
  /// @param tolength outparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray64_rpad_length_axis1(
      int64_t* tooffsets,
      const int64_t* fromoffsets,
      int64_t fromlength,
      int64_t target,
      int64_t* tolength);

  /// @param toindex outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param fromlength inparam
  /// @param target inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray32_rpad_axis1_64(
      int64_t* toindex,
      const int32_t* fromoffsets,
      int64_t fromlength,
      int64_t target);
  /// @param toindex outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param fromlength inparam
  /// @param target inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArrayU32_rpad_axis1_64(
      int64_t* toindex,
      const uint32_t* fromoffsets,
      int64_t fromlength,
      int64_t target);
  /// @param toindex outparam
  /// @param fromoffsets inparam role: ListOffsetArray-offsets
  /// @param fromlength inparam
  /// @param target inparam
  EXPORT_SYMBOL struct Error
    awkward_ListOffsetArray64_rpad_axis1_64(
      int64_t* toindex,
      const int64_t* fromoffsets,
      int64_t fromlength,
      int64_t target);

  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_validity(
      const int32_t* starts,
      const int32_t* stops,
      int64_t length,
      int64_t lencontent);
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_validity(
      const uint32_t* starts,
      const uint32_t* stops,
      int64_t length,
      int64_t lencontent);
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  /// @param lencontent inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_validity(
      const int64_t* starts,
      const int64_t* stops,
      int64_t length,
      int64_t lencontent);

  /// @param index inparam role: IndexedArray-index
  /// @param length inparam
  /// @param lencontent inparam role: IndexedArray-length
  /// @param isoption inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray32_validity(
      const int32_t* index,
      int64_t length,
      int64_t lencontent,
      bool isoption);
  /// @param index inparam role: IndexedArray-index
  /// @param length inparam
  /// @param lencontent inparam role: IndexedArray-length
  /// @param isoption inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_IndexedArrayU32_validity(
      const uint32_t* index,
      int64_t length,
      int64_t lencontent,
      bool isoption);
  /// @param index inparam role: IndexedArray-index
  /// @param length inparam
  /// @param lencontent inparam role: IndexedArray-length
  /// @param isoption inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_IndexedArray64_validity(
      const int64_t* index,
      int64_t length,
      int64_t lencontent,
      bool isoption);

  /// @param tags inparam role: UnionArray-tags
  /// @param index inparam role: IndexedArray-index
  /// @param length inparam
  /// @param numcontents inparam role: UnionArray-length
  /// @param lencontents inparam role: ListOffsetArray-offsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_32_validity(
      const int8_t* tags,
      const int32_t* index,
      int64_t length,
      int64_t numcontents,
      const int64_t* lencontents);
  /// @param tags inparam role: UnionArray-tags
  /// @param index inparam role: IndexedArray-index
  /// @param length inparam
  /// @param numcontents inparam role: UnionArray-length
  /// @param lencontents inparam role: ListOffsetArray-offsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_U32_validity(
      const int8_t* tags,
      const uint32_t* index,
      int64_t length,
      int64_t numcontents,
      const int64_t* lencontents);
  /// @param tags inparam role: UnionArray-tags
  /// @param index inparam role: IndexedArray-index
  /// @param length inparam
  /// @param numcontents inparam role: UnionArray-length
  /// @param lencontents inparam role: ListOffsetArray-offsets
  EXPORT_SYMBOL struct Error
    awkward_UnionArray8_64_validity(
      const int8_t* tags,
      const int64_t* index,
      int64_t length,
      int64_t numcontents,
      const int64_t* lencontents);

  /// @param toindex outparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_localindex_64(
      int64_t* toindex,
      int64_t length);

  /// @param toindex outparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_localindex_64(
      int64_t* toindex,
      const int32_t* offsets,
      int64_t length);
  /// @param toindex outparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_localindex_64(
      int64_t* toindex,
      const uint32_t* offsets,
      int64_t length);
  /// @param toindex outparam
  /// @param offsets inparam role: ListOffsetArray-offsets
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_localindex_64(
      int64_t* toindex,
      const int64_t* offsets,
      int64_t length);

  /// @param toindex outparam
  /// @param size inparam role: RegularArray-size
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_localindex_64(
      int64_t* toindex,
      int64_t size,
      int64_t length);

  /// @param toindex outparam
  /// @param n inparam
  /// @param replacement inparam
  /// @param singlelen inparam
  EXPORT_SYMBOL struct Error
    awkward_combinations_64(
      int64_t* toindex,
      int64_t n,
      bool replacement,
      int64_t singlelen);

  /// @param totallen outparam
  /// @param tooffsets outparam
  /// @param n inparam
  /// @param replacement inparam role: ByteMaskedArray-valid_when
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_combinations_length_64(
      int64_t* totallen,
      int64_t* tooffsets,
      int64_t n,
      bool replacement,
      const int32_t* starts,
      const int32_t* stops,
      int64_t length);
  /// @param totallen outparam
  /// @param tooffsets outparam
  /// @param n inparam
  /// @param replacement inparam role: ByteMaskedArray-valid_when
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_combinations_length_64(
      int64_t* totallen,
      int64_t* tooffsets,
      int64_t n,
      bool replacement,
      const uint32_t* starts,
      const uint32_t* stops,
      int64_t length);
  /// @param totallen outparam
  /// @param tooffsets outparam
  /// @param n inparam
  /// @param replacement inparam role: ByteMaskedArray-valid_when
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_combinations_length_64(
      int64_t* totallen,
      int64_t* tooffsets,
      int64_t n,
      bool replacement,
      const int64_t* starts,
      const int64_t* stops,
      int64_t length);

  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam
  /// @param n inparam
  /// @param replacement inparam
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray32_combinations_64(
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t n,
      bool replacement,
      const int32_t* starts,
      const int32_t* stops,
      int64_t length);
  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam
  /// @param n inparam
  /// @param replacement inparam
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArrayU32_combinations_64(
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t n,
      bool replacement,
      const uint32_t* starts,
      const uint32_t* stops,
      int64_t length);
  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam
  /// @param n inparam
  /// @param replacement inparam
  /// @param starts inparam role: ListArray-starts
  /// @param stops inparam role: ListArray-stops
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_ListArray64_combinations_64(
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t n,
      bool replacement,
      const int64_t* starts,
      const int64_t* stops,
      int64_t length);

  /// @param tocarry outparam
  /// @param toindex outparam
  /// @param fromindex inparam
  /// @param n inparam
  /// @param replacement inparam
  /// @param size inparam
  /// @param length inparam
  EXPORT_SYMBOL struct Error
    awkward_RegularArray_combinations_64(
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t n,
      bool replacement,
      int64_t size,
      int64_t length);

  /// @param tomask outparam
  /// @param theirmask inparam role: ByteMaskedArray-mask
  /// @param mymask inparam role: ByteMaskedArray2-mask
  /// @param length inparam
  /// @param validwhen inparam role: ByteMaskedArray-valid_when
  EXPORT_SYMBOL struct Error
    awkward_ByteMaskedArray_overlay_mask8(
      int8_t* tomask,
      const int8_t* theirmask,
      const int8_t* mymask,
      int64_t length,
      bool validwhen);

  /// @param tobytemask outparam
  /// @param frombitmask inparam role: BitMaskedArray-mask
  /// @param bitmasklength inparam
  /// @param validwhen inparam role: BitMaskedArray-valid_when
  /// @param lsb_order inparam role: BitMaskedArray-lsb_order
  EXPORT_SYMBOL struct Error
    awkward_BitMaskedArray_to_ByteMaskedArray(
      int8_t* tobytemask,
      const uint8_t* frombitmask,
      int64_t bitmasklength,
      bool validwhen,
      bool lsb_order);

  /// @param toindex outparam
  /// @param frombitmask inparam role: BitMaskedArray-mask
  /// @param bitmasklength inparam
  /// @param validwhen inparam role: BitMaskedArray-valid_when
  /// @param lsb_order inparam role: BitMaskedArray-lsb_order
  EXPORT_SYMBOL struct Error
    awkward_BitMaskedArray_to_IndexedOptionArray64(
      int64_t* toindex,
      const uint8_t* frombitmask,
      int64_t bitmasklength,
      bool validwhen,
      bool lsb_order);

}

#endif // AWKWARD_KERNELS_OPERATIONS_H_
