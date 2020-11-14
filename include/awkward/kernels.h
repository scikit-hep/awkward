// AUTO GENERATED: DO NOT EDIT BY HAND!
// To regenerate file, execute - python dev/generate-kernelheader.py

#ifndef AWKWARD_KERNELS_H_
#define AWKWARD_KERNELS_H_

#include "awkward/common.h"

extern "C" {
  EXPORT_SYMBOL ERROR
  awkward_BitMaskedArray_to_ByteMaskedArray(
    int8_t* tobytemask,
    const uint8_t* frombitmask,
    int64_t bitmasklength,
    bool validwhen,
    bool lsb_order);

  EXPORT_SYMBOL ERROR
  awkward_BitMaskedArray_to_IndexedOptionArray64(
    int64_t* toindex,
    const uint8_t* frombitmask,
    int64_t bitmasklength,
    bool validwhen,
    bool lsb_order);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_getitem_carry_64(
    int8_t* tomask,
    const int8_t* frommask,
    int64_t lenmask,
    const int64_t* fromcarry,
    int64_t lencarry);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_getitem_nextcarry_64(
    int64_t* tocarry,
    const int8_t* mask,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    int64_t* outindex,
    const int8_t* mask,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_mask8(
    int8_t* tomask,
    const int8_t* frommask,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_numnull(
    int64_t* numnull,
    const int8_t* mask,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_overlay_mask8(
    int8_t* tomask,
    const int8_t* theirmask,
    const int8_t* mymask,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_reduce_next_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int8_t* mask,
    const int64_t* parents,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(
    int64_t* nextshifts,
    const int8_t* mask,
    int64_t length,
    bool valid_when);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
    int64_t* nextshifts,
    const int8_t* mask,
    int64_t length,
    bool valid_when,
    const int64_t* shifts);

  EXPORT_SYMBOL ERROR
  awkward_ByteMaskedArray_toIndexedOptionArray64(
    int64_t* toindex,
    const int8_t* mask,
    int64_t length,
    bool validwhen);

  EXPORT_SYMBOL ERROR
  awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
    int64_t* index_in,
    int64_t* offsets_in,
    int64_t* mask_out,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_to_Identities64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    int64_t width);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_extend(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t fromlength,
    int64_t tolength);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_extend(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t fromlength,
    int64_t tolength);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_IndexedArray32(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_IndexedArray64(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_IndexedArrayU32(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_IndexedArray32(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_IndexedArray64(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_IndexedArrayU32(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_ListArray32(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_ListArray64(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_ListArrayU32(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_ListArray32(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_ListArray64(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_ListArrayU32(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_ListOffsetArray32(
    int32_t* toptr,
    const int32_t* fromptr,
    const int32_t* fromoffsets,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_ListOffsetArray64(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* fromoffsets,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_ListOffsetArrayU32(
    int32_t* toptr,
    const int32_t* fromptr,
    const uint32_t* fromoffsets,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_ListOffsetArray32(
    int64_t* toptr,
    const int64_t* fromptr,
    const int32_t* fromoffsets,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_ListOffsetArray64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* fromoffsets,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_ListOffsetArrayU32(
    int64_t* toptr,
    const int64_t* fromptr,
    const uint32_t* fromoffsets,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_RegularArray(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t size,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_RegularArray(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t size,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_UnionArray8_32(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_UnionArray8_64(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_Identities32_from_UnionArray8_U32(
    bool* uniquecontents,
    int32_t* toptr,
    const int32_t* fromptr,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_UnionArray8_32(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_UnionArray8_64(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_from_UnionArray8_U32(
    bool* uniquecontents,
    int64_t* toptr,
    const int64_t* fromptr,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t tolength,
    int64_t fromlength,
    int64_t fromwidth,
    int64_t which);

  EXPORT_SYMBOL ERROR
  awkward_Identities32_getitem_carry_64(
    int32_t* newidentitiesptr,
    const int32_t* identitiesptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t width,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_Identities64_getitem_carry_64(
    int64_t* newidentitiesptr,
    const int64_t* identitiesptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t width,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_Index32_to_Index64(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_Index8_to_Index64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexU32_to_Index64(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexU8_to_Index64(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_fill_to64_from32(
    int64_t* toindex,
    int64_t toindexoffset,
    const int32_t* fromindex,
    int64_t length,
    int64_t base);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_fill_to64_from64(
    int64_t* toindex,
    int64_t toindexoffset,
    const int64_t* fromindex,
    int64_t length,
    int64_t base);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_fill_to64_fromU32(
    int64_t* toindex,
    int64_t toindexoffset,
    const uint32_t* fromindex,
    int64_t length,
    int64_t base);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_fill_to64_count(
    int64_t* toindex,
    int64_t toindexoffset,
    int64_t length,
    int64_t base);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_flatten_nextcarry_64(
    int64_t* tocarry,
    const int32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_flatten_nextcarry_64(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_flatten_nextcarry_64(
    int64_t* tocarry,
    const uint32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_flatten_none2empty_64(
    int64_t* outoffsets,
    const int32_t* outindex,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetslength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_flatten_none2empty_64(
    int64_t* outoffsets,
    const int64_t* outindex,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetslength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_flatten_none2empty_64(
    int64_t* outoffsets,
    const uint32_t* outindex,
    int64_t outindexlength,
    const int64_t* offsets,
    int64_t offsetslength);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_getitem_adjust_outindex_64(
    int8_t* tomask,
    int64_t* toindex,
    int64_t* tononzero,
    const int64_t* fromindex,
    int64_t fromindexlength,
    const int64_t* nonzero,
    int64_t nonzerolength);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_getitem_carry_64(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* fromcarry,
    int64_t lenindex,
    int64_t lencarry);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_getitem_carry_64(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* fromcarry,
    int64_t lenindex,
    int64_t lencarry);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_getitem_carry_64(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* fromcarry,
    int64_t lenindex,
    int64_t lencarry);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_getitem_nextcarry_64(
    int64_t* tocarry,
    const int32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_getitem_nextcarry_64(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_getitem_nextcarry_64(
    int64_t* tocarry,
    const uint32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    int32_t* toindex,
    const int32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_getitem_nextcarry_outindex_64(
    int64_t* tocarry,
    uint32_t* toindex,
    const uint32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_getitem_nextcarry_outindex_mask_64(
    int64_t* tocarry,
    int64_t* toindex,
    const int32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_getitem_nextcarry_outindex_mask_64(
    int64_t* tocarry,
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_getitem_nextcarry_outindex_mask_64(
    int64_t* tocarry,
    int64_t* toindex,
    const uint32_t* fromindex,
    int64_t lenindex,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* starts,
    const int64_t* parents,
    const int64_t parentslength,
    const int64_t* nextparents,
    const int64_t nextlen);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_mask8(
    int8_t* tomask,
    const int32_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_mask8(
    int8_t* tomask,
    const int64_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_mask8(
    int8_t* tomask,
    const uint32_t* fromindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_numnull(
    int64_t* numnull,
    const int32_t* fromindex,
    int64_t lenindex);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_numnull(
    int64_t* numnull,
    const int64_t* fromindex,
    int64_t lenindex);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_numnull(
    int64_t* numnull,
    const uint32_t* fromindex,
    int64_t lenindex);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_overlay_mask8_to64(
    int64_t* toindex,
    const int8_t* mask,
    const int32_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_overlay_mask8_to64(
    int64_t* toindex,
    const int8_t* mask,
    const int64_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_overlay_mask8_to64(
    int64_t* toindex,
    const int8_t* mask,
    const uint32_t* fromindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_reduce_next_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int32_t* index,
    int64_t* parents,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_reduce_next_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const int64_t* index,
    int64_t* parents,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_reduce_next_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t* outindex,
    const uint32_t* index,
    int64_t* parents,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray_reduce_next_fix_offsets_64(
    int64_t* outoffsets,
    const int64_t* starts,
    int64_t startslength,
    int64_t outindexlength);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_64(
    int64_t* nextshifts,
    const int32_t* index,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_64(
    int64_t* nextshifts,
    const int64_t* index,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_64(
    int64_t* nextshifts,
    const uint32_t* index,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_reduce_next_nonlocal_nextshifts_fromshifts_64(
    int64_t* nextshifts,
    const int32_t* index,
    int64_t length,
    const int64_t* shifts);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_reduce_next_nonlocal_nextshifts_fromshifts_64(
    int64_t* nextshifts,
    const int64_t* index,
    int64_t length,
    const int64_t* shifts);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_reduce_next_nonlocal_nextshifts_fromshifts_64(
    int64_t* nextshifts,
    const uint32_t* index,
    int64_t length,
    const int64_t* shifts);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_simplify32_to64(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_simplify64_to64(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_simplifyU32_to64(
    int64_t* toindex,
    const int32_t* outerindex,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_simplify32_to64(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_simplify64_to64(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_simplifyU32_to64(
    int64_t* toindex,
    const int64_t* outerindex,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_simplify32_to64(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outerlength,
    const int32_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_simplify64_to64(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outerlength,
    const int64_t* innerindex,
    int64_t innerlength);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_simplifyU32_to64(
    int64_t* toindex,
    const uint32_t* outerindex,
    int64_t outerlength,
    const uint32_t* innerindex,
    int64_t innerlength);

  EXPORT_SYMBOL ERROR
  awkward_IndexedArray32_validity(
    const int32_t* index,
    int64_t length,
    int64_t lencontent,
    bool isoption);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArray64_validity(
    const int64_t* index,
    int64_t length,
    int64_t lencontent,
    bool isoption);
  EXPORT_SYMBOL ERROR
  awkward_IndexedArrayU32_validity(
    const uint32_t* index,
    int64_t length,
    int64_t lencontent,
    bool isoption);

  EXPORT_SYMBOL ERROR
  awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64(
    int64_t* toindex,
    const int8_t* frommask,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_broadcast_tooffsets_64(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetslength,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_broadcast_tooffsets_64(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetslength,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_broadcast_tooffsets_64(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetslength,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_combinations_64(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const int32_t* starts,
    const int32_t* stops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_combinations_64(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const int64_t* starts,
    const int64_t* stops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_combinations_64(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    const uint32_t* starts,
    const uint32_t* stops,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_combinations_length_64(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const int32_t* starts,
    const int32_t* stops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_combinations_length_64(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const int64_t* starts,
    const int64_t* stops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_combinations_length_64(
    int64_t* totallen,
    int64_t* tooffsets,
    int64_t n,
    bool replacement,
    const uint32_t* starts,
    const uint32_t* stops,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_compact_offsets_64(
    int64_t* tooffsets,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_compact_offsets_64(
    int64_t* tooffsets,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_compact_offsets_64(
    int64_t* tooffsets,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray_fill_to64_from32(
    int64_t* tostarts,
    int64_t tostartsoffset,
    int64_t* tostops,
    int64_t tostopsoffset,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t length,
    int64_t base);
  EXPORT_SYMBOL ERROR
  awkward_ListArray_fill_to64_from64(
    int64_t* tostarts,
    int64_t tostartsoffset,
    int64_t* tostops,
    int64_t tostopsoffset,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length,
    int64_t base);
  EXPORT_SYMBOL ERROR
  awkward_ListArray_fill_to64_fromU32(
    int64_t* tostarts,
    int64_t tostartsoffset,
    int64_t* tostops,
    int64_t tostopsoffset,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t length,
    int64_t base);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_carry_64(
    int32_t* tostarts,
    int32_t* tostops,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromcarry,
    int64_t lenstarts,
    int64_t lencarry);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_carry_64(
    int64_t* tostarts,
    int64_t* tostops,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromcarry,
    int64_t lenstarts,
    int64_t lencarry);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_carry_64(
    uint32_t* tostarts,
    uint32_t* tostops,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromcarry,
    int64_t lenstarts,
    int64_t lencarry);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_jagged_apply_64(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen,
    const int64_t* sliceindex,
    int64_t sliceinnerlen,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t contentlen);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_jagged_apply_64(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen,
    const int64_t* sliceindex,
    int64_t sliceinnerlen,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t contentlen);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_jagged_apply_64(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen,
    const int64_t* sliceindex,
    int64_t sliceinnerlen,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t contentlen);

  EXPORT_SYMBOL ERROR
  awkward_ListArray_getitem_jagged_carrylen_64(
    int64_t* carrylen,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_jagged_descend_64(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen,
    const int32_t* fromstarts,
    const int32_t* fromstops);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_jagged_descend_64(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen,
    const int64_t* fromstarts,
    const int64_t* fromstops);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_jagged_descend_64(
    int64_t* tooffsets,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t sliceouterlen,
    const uint32_t* fromstarts,
    const uint32_t* fromstops);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t jaggedsize,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t jaggedsize,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t jaggedsize,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray_getitem_jagged_numvalid_64(
    int64_t* numvalid,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t length,
    const int64_t* missing,
    int64_t missinglength);

  EXPORT_SYMBOL ERROR
  awkward_ListArray_getitem_jagged_shrink_64(
    int64_t* tocarry,
    int64_t* tosmalloffsets,
    int64_t* tolargeoffsets,
    const int64_t* slicestarts,
    const int64_t* slicestops,
    int64_t length,
    const int64_t* missing);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_array_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromarray,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_array_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromarray,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_array_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromarray,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_array_advanced_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_array_advanced_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_array_advanced_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    const int64_t* fromarray,
    const int64_t* fromadvanced,
    int64_t lenstarts,
    int64_t lenarray,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_at_64(
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t at);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_at_64(
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t at);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_at_64(
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t at);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_range_64(
    int32_t* tooffsets,
    int64_t* tocarry,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_range_64(
    int64_t* tooffsets,
    int64_t* tocarry,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_range_64(
    uint32_t* tooffsets,
    int64_t* tocarry,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_range_carrylength(
    int64_t* carrylength,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_range_carrylength(
    int64_t* carrylength,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_range_carrylength(
    int64_t* carrylength,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_range_counts_64(
    int64_t* total,
    const int32_t* fromoffsets,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_range_counts_64(
    int64_t* total,
    const int64_t* fromoffsets,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_range_counts_64(
    int64_t* total,
    const uint32_t* fromoffsets,
    int64_t lenstarts);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_getitem_next_range_spreadadvanced_64(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int32_t* fromoffsets,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_getitem_next_range_spreadadvanced_64(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int64_t* fromoffsets,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_getitem_next_range_spreadadvanced_64(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const uint32_t* fromoffsets,
    int64_t lenstarts);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_localindex_64(
    int64_t* toindex,
    const int32_t* offsets,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_localindex_64(
    int64_t* toindex,
    const int64_t* offsets,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_localindex_64(
    int64_t* toindex,
    const uint32_t* offsets,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_min_range(
    int64_t* tomin,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_min_range(
    int64_t* tomin,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_min_range(
    int64_t* tomin,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t lenstarts);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_num_64(
    int64_t* tonum,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_num_64(
    int64_t* tonum,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_num_64(
    int64_t* tonum,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_rpad_and_clip_length_axis1(
    int64_t* tomin,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int64_t target,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_rpad_and_clip_length_axis1(
    int64_t* tomin,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t target,
    int64_t lenstarts);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_rpad_and_clip_length_axis1(
    int64_t* tomin,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    int64_t target,
    int64_t lenstarts);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_rpad_axis1_64(
    int64_t* toindex,
    const int32_t* fromstarts,
    const int32_t* fromstops,
    int32_t* tostarts,
    int32_t* tostops,
    int64_t target,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_rpad_axis1_64(
    int64_t* toindex,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t* tostarts,
    int64_t* tostops,
    int64_t target,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_rpad_axis1_64(
    int64_t* toindex,
    const uint32_t* fromstarts,
    const uint32_t* fromstops,
    uint32_t* tostarts,
    uint32_t* tostops,
    int64_t target,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListArray32_validity(
    const int32_t* starts,
    const int32_t* stops,
    int64_t length,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArray64_validity(
    const int64_t* starts,
    const int64_t* stops,
    int64_t length,
    int64_t lencontent);
  EXPORT_SYMBOL ERROR
  awkward_ListArrayU32_validity(
    const uint32_t* starts,
    const uint32_t* stops,
    int64_t length,
    int64_t lencontent);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray32_compact_offsets_64(
    int64_t* tooffsets,
    const int32_t* fromoffsets,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_compact_offsets_64(
    int64_t* tooffsets,
    const int64_t* fromoffsets,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArrayU32_compact_offsets_64(
    int64_t* tooffsets,
    const uint32_t* fromoffsets,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray32_flatten_offsets_64(
    int64_t* tooffsets,
    const int32_t* outeroffsets,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetslen);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_flatten_offsets_64(
    int64_t* tooffsets,
    const int64_t* outeroffsets,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetslen);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArrayU32_flatten_offsets_64(
    int64_t* tooffsets,
    const uint32_t* outeroffsets,
    int64_t outeroffsetslen,
    const int64_t* inneroffsets,
    int64_t inneroffsetslen);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_merge_offsets_64(
    int64_t* tooffsets,
    const int64_t* fromoffsets,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_getitem_adjust_offsets_64(
    int64_t* tooffsets,
    int64_t* tononzero,
    const int64_t* fromoffsets,
    int64_t length,
    const int64_t* nonzero,
    int64_t nonzerolength);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_getitem_adjust_offsets_index_64(
    int64_t* tooffsets,
    int64_t* tononzero,
    const int64_t* fromoffsets,
    int64_t length,
    const int64_t* index,
    int64_t indexlength,
    const int64_t* nonzero,
    int64_t nonzerolength,
    const int8_t* originalmask,
    int64_t masklength);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_local_preparenext_64(
    int64_t* tocarry,
    const int64_t* fromindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_global_startstop_64(
    int64_t* globalstart,
    int64_t* globalstop,
    const int64_t* offsets,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_local_nextparents_64(
    int64_t* nextparents,
    const int64_t* offsets,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_local_outoffsets_64(
    int64_t* outoffsets,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_nonlocal_findgaps_64(
    int64_t* gaps,
    const int64_t* parents,
    int64_t lenparents);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
    int64_t* maxcount,
    int64_t* offsetscopy,
    const int64_t* offsets,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64(
    int64_t* nummissing,
    int64_t* missing,
    int64_t* nextshifts,
    const int64_t* offsets,
    int64_t length,
    const int64_t* starts,
    const int64_t* parents,
    int64_t maxcount,
    int64_t nextlen,
    const int64_t* nextcarry);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
    int64_t* nextstarts,
    const int64_t* nextparents,
    int64_t nextlen);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64(
    int64_t* outstarts,
    int64_t* outstops,
    const int64_t* distincts,
    int64_t lendistincts,
    const int64_t* gaps,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray_reduce_nonlocal_preparenext_64(
    int64_t* nextcarry,
    int64_t* nextparents,
    int64_t nextlen,
    int64_t* maxnextparents,
    int64_t* distincts,
    int64_t distinctslen,
    int64_t* offsetscopy,
    const int64_t* offsets,
    int64_t length,
    const int64_t* parents,
    int64_t maxcount);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray32_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const int32_t* fromoffsets,
    int64_t length,
    int64_t target);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const int64_t* fromoffsets,
    int64_t length,
    int64_t target);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArrayU32_rpad_and_clip_axis1_64(
    int64_t* toindex,
    const uint32_t* fromoffsets,
    int64_t length,
    int64_t target);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray32_rpad_axis1_64(
    int64_t* toindex,
    const int32_t* fromoffsets,
    int64_t fromlength,
    int64_t target);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_rpad_axis1_64(
    int64_t* toindex,
    const int64_t* fromoffsets,
    int64_t fromlength,
    int64_t target);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArrayU32_rpad_axis1_64(
    int64_t* toindex,
    const uint32_t* fromoffsets,
    int64_t fromlength,
    int64_t target);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray32_rpad_length_axis1(
    int32_t* tooffsets,
    const int32_t* fromoffsets,
    int64_t fromlength,
    int64_t target,
    int64_t* tolength);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_rpad_length_axis1(
    int64_t* tooffsets,
    const int64_t* fromoffsets,
    int64_t fromlength,
    int64_t target,
    int64_t* tolength);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArrayU32_rpad_length_axis1(
    uint32_t* tooffsets,
    const uint32_t* fromoffsets,
    int64_t fromlength,
    int64_t target,
    int64_t* tolength);

  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray32_toRegularArray(
    int64_t* size,
    const int32_t* fromoffsets,
    int64_t offsetslength);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArray64_toRegularArray(
    int64_t* size,
    const int64_t* fromoffsets,
    int64_t offsetslength);
  EXPORT_SYMBOL ERROR
  awkward_ListOffsetArrayU32_toRegularArray(
    int64_t* size,
    const uint32_t* fromoffsets,
    int64_t offsetslength);

  EXPORT_SYMBOL ERROR
  awkward_MaskedArray32_getitem_next_jagged_project(
    int32_t* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_MaskedArray64_getitem_next_jagged_project(
    int64_t* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_MaskedArrayU32_getitem_next_jagged_project(
    uint32_t* index,
    int64_t* starts_in,
    int64_t* stops_in,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_copy(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_contiguous_copy_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len,
    int64_t stride,
    const int64_t* pos);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_contiguous_init_64(
    int64_t* toptr,
    int64_t skip,
    int64_t stride);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_contiguous_next_64(
    int64_t* topos,
    const int64_t* frompos,
    int64_t length,
    int64_t skip,
    int64_t stride);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromint8(
    int8_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromint16(
    int8_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromint32(
    int8_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromint64(
    int8_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromuint8(
    int8_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromuint16(
    int8_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromuint32(
    int8_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromuint64(
    int8_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromfloat32(
    int8_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_fromfloat64(
    int8_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromint8(
    int16_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromint16(
    int16_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromint32(
    int16_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromint64(
    int16_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromuint8(
    int16_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromuint16(
    int16_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromuint32(
    int16_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromuint64(
    int16_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromfloat32(
    int16_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_fromfloat64(
    int16_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromint8(
    int32_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromint16(
    int32_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromint32(
    int32_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromint64(
    int32_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromuint8(
    int32_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromuint16(
    int32_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromuint32(
    int32_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromuint64(
    int32_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromfloat32(
    int32_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_fromfloat64(
    int32_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromint8(
    int64_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromint16(
    int64_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromint32(
    int64_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromint64(
    int64_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromuint8(
    int64_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromuint16(
    int64_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromuint32(
    int64_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromuint64(
    int64_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromfloat32(
    int64_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_fromfloat64(
    int64_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromint8(
    uint8_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromint16(
    uint8_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromint32(
    uint8_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromint64(
    uint8_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromuint8(
    uint8_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromuint16(
    uint8_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromuint32(
    uint8_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromuint64(
    uint8_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromfloat32(
    uint8_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_fromfloat64(
    uint8_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromint8(
    uint16_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromint16(
    uint16_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromint32(
    uint16_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromint64(
    uint16_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromuint8(
    uint16_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromuint16(
    uint16_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromuint32(
    uint16_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromuint64(
    uint16_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromfloat32(
    uint16_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_fromfloat64(
    uint16_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromint8(
    uint32_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromint16(
    uint32_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromint32(
    uint32_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromint64(
    uint32_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromuint8(
    uint32_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromuint16(
    uint32_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromuint32(
    uint32_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromuint64(
    uint32_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromfloat32(
    uint32_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_fromfloat64(
    uint32_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromint8(
    uint64_t* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromint16(
    uint64_t* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromint32(
    uint64_t* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromint64(
    uint64_t* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromuint8(
    uint64_t* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromuint16(
    uint64_t* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromuint32(
    uint64_t* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromuint64(
    uint64_t* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromfloat32(
    uint64_t* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_fromfloat64(
    uint64_t* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromint8(
    float* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromint16(
    float* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromint32(
    float* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromint64(
    float* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromuint8(
    float* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromuint16(
    float* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromuint32(
    float* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromuint64(
    float* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromfloat32(
    float* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_fromfloat64(
    float* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromint8(
    double* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromint16(
    double* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromint32(
    double* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromint64(
    double* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromuint8(
    double* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromuint16(
    double* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromuint32(
    double* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromuint64(
    double* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromfloat32(
    double* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_fromfloat64(
    double* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_frombool(
    bool* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint8_frombool(
    int8_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint16_frombool(
    int16_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint32_frombool(
    int32_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_toint64_frombool(
    int64_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint8_frombool(
    uint8_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint16_frombool(
    uint16_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint32_frombool(
    uint32_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_touint64_frombool(
    uint64_t* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat32_frombool(
    float* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tofloat64_frombool(
    double* toptr,
    int64_t tooffset,
    const bool* fromptr,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromint8(
    bool* toptr,
    int64_t tooffset,
    const int8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromint16(
    bool* toptr,
    int64_t tooffset,
    const int16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromint32(
    bool* toptr,
    int64_t tooffset,
    const int32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromint64(
    bool* toptr,
    int64_t tooffset,
    const int64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromuint8(
    bool* toptr,
    int64_t tooffset,
    const uint8_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromuint16(
    bool* toptr,
    int64_t tooffset,
    const uint16_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromuint32(
    bool* toptr,
    int64_t tooffset,
    const uint32_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromuint64(
    bool* toptr,
    int64_t tooffset,
    const uint64_t* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromfloat32(
    bool* toptr,
    int64_t tooffset,
    const float* fromptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_fill_tobool_fromfloat64(
    bool* toptr,
    int64_t tooffset,
    const double* fromptr,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_boolean_nonzero_64(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    int64_t stride);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_boolean_numtrue(
    int64_t* numtrue,
    const int8_t* fromptr,
    int64_t length,
    int64_t stride);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_next_array_64(
    int64_t* nextcarryptr,
    int64_t* nextadvancedptr,
    const int64_t* carryptr,
    const int64_t* flatheadptr,
    int64_t lencarry,
    int64_t lenflathead,
    int64_t skip);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_next_array_advanced_64(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    const int64_t* advancedptr,
    const int64_t* flatheadptr,
    int64_t lencarry,
    int64_t skip);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_next_at_64(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t skip,
    int64_t at);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_next_null_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t len,
    int64_t stride,
    const int64_t* pos);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_getitem_next_range_64(
    int64_t* nextcarryptr,
    const int64_t* carryptr,
    int64_t lencarry,
    int64_t lenhead,
    int64_t skip,
    int64_t start,
    int64_t step);

  EXPORT_SYMBOL ERROR
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

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_reduce_adjust_starts_64(
    int64_t* toptr,
    int64_t outlength,
    const int64_t* parents,
    const int64_t* starts);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_reduce_adjust_starts_shifts_64(
    int64_t* toptr,
    int64_t outlength,
    const int64_t* parents,
    const int64_t* starts,
    const int64_t* shifts);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
    int8_t* toptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_NumpyArray_sort_asstrings_uint8(
    uint8_t* toptr,
    const uint8_t* fromptr,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t* outoffsets,
    bool ascending,
    bool stable);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_broadcast_tooffsets_64(
    const int64_t* fromoffsets,
    int64_t offsetslength,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_broadcast_tooffsets_size1_64(
    int64_t* tocarry,
    const int64_t* fromoffsets,
    int64_t offsetslength);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_combinations_64(
    int64_t** tocarry,
    int64_t* toindex,
    int64_t* fromindex,
    int64_t n,
    bool replacement,
    int64_t size,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_compact_offsets64(
    int64_t* tooffsets,
    int64_t length,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_carry_64(
    int64_t* tocarry,
    const int64_t* fromcarry,
    int64_t lencarry,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_jagged_expand_64(
    int64_t* multistarts,
    int64_t* multistops,
    const int64_t* singleoffsets,
    int64_t regularsize,
    int64_t regularlength);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_next_array_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromarray,
    int64_t length,
    int64_t lenarray,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_next_array_advanced_64(
    int64_t* tocarry,
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    const int64_t* fromarray,
    int64_t length,
    int64_t lenarray,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_next_array_regularize_64(
    int64_t* toarray,
    const int64_t* fromarray,
    int64_t lenarray,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_next_at_64(
    int64_t* tocarry,
    int64_t at,
    int64_t length,
    int64_t size);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_next_range_64(
    int64_t* tocarry,
    int64_t regular_start,
    int64_t step,
    int64_t length,
    int64_t size,
    int64_t nextsize);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_getitem_next_range_spreadadvanced_64(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    int64_t length,
    int64_t nextsize);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_localindex_64(
    int64_t* toindex,
    int64_t size,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_num_64(
    int64_t* tonum,
    int64_t size,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_RegularArray_rpad_and_clip_axis1_64(
    int64_t* toindex,
    int64_t target,
    int64_t size,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillindex_to64_from32(
    int64_t* toindex,
    int64_t toindexoffset,
    const int32_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillindex_to64_from64(
    int64_t* toindex,
    int64_t toindexoffset,
    const int64_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillindex_to64_fromU32(
    int64_t* toindex,
    int64_t toindexoffset,
    const uint32_t* fromindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillindex_to64_count(
    int64_t* toindex,
    int64_t toindexoffset,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillna_from32_to64(
    int64_t* toindex,
    const int32_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillna_from64_to64(
    int64_t* toindex,
    const int64_t* fromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray_fillna_fromU32_to64(
    int64_t* toindex,
    const uint32_t* fromindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray_filltags_to8_from8(
    int8_t* totags,
    int64_t totagsoffset,
    const int8_t* fromtags,
    int64_t length,
    int64_t base);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray_filltags_to8_const(
    int8_t* totags,
    int64_t totagsoffset,
    int64_t length,
    int64_t base);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray_mergetags_to8_const(
    int8_t* totags,
    int64_t* toindex,
    int64_t offset,
    const int64_t* fromoffsets,
    int64_t index,
    int8_t tag,
    int64_t nextoffset);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray32_flatten_combine_64(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t length,
    int64_t** offsetsraws);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray64_flatten_combine_64(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t length,
    int64_t** offsetsraws);
  EXPORT_SYMBOL ERROR
  awkward_UnionArrayU32_flatten_combine_64(
    int8_t* totags,
    int64_t* toindex,
    int64_t* tooffsets,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t length,
    int64_t** offsetsraws);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray32_flatten_length_64(
    int64_t* total_length,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t length,
    int64_t** offsetsraws);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray64_flatten_length_64(
    int64_t* total_length,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t length,
    int64_t** offsetsraws);
  EXPORT_SYMBOL ERROR
  awkward_UnionArrayU32_flatten_length_64(
    int64_t* total_length,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t length,
    int64_t** offsetsraws);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_32_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t length,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_64_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t length,
    int64_t which);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_U32_project_64(
    int64_t* lenout,
    int64_t* tocarry,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t length,
    int64_t which);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_32_regular_index(
    int32_t* toindex,
    int32_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_64_regular_index(
    int64_t* toindex,
    int64_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_U32_regular_index(
    uint32_t* toindex,
    uint32_t* current,
    int64_t size,
    const int8_t* fromtags,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_regular_index_getsize(
    int64_t* size,
    const int8_t* fromtags,
    int64_t length);

  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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
  EXPORT_SYMBOL ERROR
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

  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_32_simplify_one_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* fromtags,
    const int32_t* fromindex,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_64_simplify_one_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* fromtags,
    const int64_t* fromindex,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_U32_simplify_one_to8_64(
    int8_t* totags,
    int64_t* toindex,
    const int8_t* fromtags,
    const uint32_t* fromindex,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base);

  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_32_validity(
    const int8_t* tags,
    const int32_t* index,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_64_validity(
    const int8_t* tags,
    const int64_t* index,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents);
  EXPORT_SYMBOL ERROR
  awkward_UnionArray8_U32_validity(
    const int8_t* tags,
    const uint32_t* index,
    int64_t length,
    int64_t numcontents,
    const int64_t* lencontents);

  EXPORT_SYMBOL ERROR
  awkward_argsort_bool(
    int64_t* toptr,
    const bool* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_int8(
    int64_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_int16(
    int64_t* toptr,
    const int16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_int32(
    int64_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_int64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_uint8(
    int64_t* toptr,
    const uint8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_uint16(
    int64_t* toptr,
    const uint16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_uint32(
    int64_t* toptr,
    const uint32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_uint64(
    int64_t* toptr,
    const uint64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_float32(
    int64_t* toptr,
    const float* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_argsort_float64(
    int64_t* toptr,
    const double* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    bool ascending,
    bool stable);

  EXPORT_SYMBOL ERROR
  awkward_carry_arange32(
    int32_t* toptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_carry_arange64(
    int64_t* toptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_carry_arangeU32(
    uint32_t* toptr,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_combinations_64(
    int64_t* toindex,
    int64_t n,
    bool replacement,
    int64_t singlelen);

  EXPORT_SYMBOL ERROR
  awkward_content_reduce_zeroparents_64(
    int64_t* toparents,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_Index32_carry_64(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* carry,
    int64_t lenfromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_Index64_carry_64(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* carry,
    int64_t lenfromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_Index8_carry_64(
    int8_t* toindex,
    const int8_t* fromindex,
    const int64_t* carry,
    int64_t lenfromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexU32_carry_64(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* carry,
    int64_t lenfromindex,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexU8_carry_64(
    uint8_t* toindex,
    const uint8_t* fromindex,
    const int64_t* carry,
    int64_t lenfromindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_Index32_carry_nocheck_64(
    int32_t* toindex,
    const int32_t* fromindex,
    const int64_t* carry,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_Index64_carry_nocheck_64(
    int64_t* toindex,
    const int64_t* fromindex,
    const int64_t* carry,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_Index8_carry_nocheck_64(
    int8_t* toindex,
    const int8_t* fromindex,
    const int64_t* carry,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexU32_carry_nocheck_64(
    uint32_t* toindex,
    const uint32_t* fromindex,
    const int64_t* carry,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_IndexU8_carry_nocheck_64(
    uint8_t* toindex,
    const uint8_t* fromindex,
    const int64_t* carry,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_index_rpad_and_clip_axis0_64(
    int64_t* toindex,
    int64_t target,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_index_rpad_and_clip_axis1_64(
    int64_t* tostarts,
    int64_t* tostops,
    int64_t target,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_localindex_64(
    int64_t* toindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_missing_repeat_64(
    int64_t* outindex,
    const int64_t* index,
    int64_t indexlength,
    int64_t repetitions,
    int64_t regularsize);

  EXPORT_SYMBOL ERROR
  awkward_new_Identities32(
    int32_t* toptr,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_new_Identities64(
    int64_t* toptr,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_uint8_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_uint16_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_uint32_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_uint64_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_float32_64(
    int64_t* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_float64_64(
    int64_t* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_argmax_bool_64(
    int64_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_uint8_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_uint16_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_uint32_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_uint64_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_float32_64(
    int64_t* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_float64_64(
    int64_t* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_argmin_bool_64(
    int64_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_count_64(
    int64_t* toptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_bool_64(
    int64_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_uint8_64(
    int64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_uint16_64(
    int64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_uint32_64(
    int64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_uint64_64(
    int64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_float32_64(
    int64_t* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_countnonzero_float64_64(
    int64_t* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_max_int8_int8_64(
    int8_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int8_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_int16_int16_64(
    int16_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int16_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_int32_int32_64(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int32_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int64_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_uint8_uint8_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint8_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_uint16_uint16_64(
    uint16_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint16_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_uint32_uint32_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_uint64_uint64_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_float32_float32_64(
    float* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    float identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_max_float64_float64_64(
    double* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    double identity);

  EXPORT_SYMBOL ERROR
  awkward_reduce_min_int8_int8_64(
    int8_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int8_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_int16_int16_64(
    int16_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int16_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_int32_int32_64(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int32_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    int64_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_uint8_uint8_64(
    uint8_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint8_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_uint16_uint16_64(
    uint16_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint16_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_uint32_uint32_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_uint64_uint64_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_float32_float32_64(
    float* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    float identity);
  EXPORT_SYMBOL ERROR
  awkward_reduce_min_float64_float64_64(
    double* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength,
    double identity);

  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int32_int8_64(
    int32_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int32_int16_64(
    int32_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int32_int32_64(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int64_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int64_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int64_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint32_uint8_64(
    uint32_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint32_uint16_64(
    uint32_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint32_uint32_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint64_uint8_64(
    uint64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint64_uint16_64(
    uint64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint64_uint32_64(
    uint64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_uint64_uint64_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_float32_float32_64(
    float* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_float64_float64_64(
    double* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_bool_64(
    bool* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_int8_64(
    bool* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_int16_64(
    bool* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_int32_64(
    bool* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_int64_64(
    bool* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_uint8_64(
    bool* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_uint16_64(
    bool* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_uint32_64(
    bool* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_uint64_64(
    bool* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_float32_64(
    bool* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_bool_float64_64(
    bool* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int32_bool_64(
    int32_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_prod_int64_bool_64(
    int64_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int32_int8_64(
    int32_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int32_int16_64(
    int32_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int32_int32_64(
    int32_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int64_int8_64(
    int64_t* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int64_int16_64(
    int64_t* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int64_int32_64(
    int64_t* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint32_uint8_64(
    uint32_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint32_uint16_64(
    uint32_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint32_uint32_64(
    uint32_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint64_uint8_64(
    uint64_t* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint64_uint16_64(
    uint64_t* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint64_uint32_64(
    uint64_t* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_uint64_uint64_64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_float32_float32_64(
    float* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_float64_float64_64(
    double* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_bool_64(
    bool* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_int8_64(
    bool* toptr,
    const int8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_int16_64(
    bool* toptr,
    const int16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_int32_64(
    bool* toptr,
    const int32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_int64_64(
    bool* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_uint8_64(
    bool* toptr,
    const uint8_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_uint16_64(
    bool* toptr,
    const uint16_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_uint32_64(
    bool* toptr,
    const uint32_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_uint64_64(
    bool* toptr,
    const uint64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_float32_64(
    bool* toptr,
    const float* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);
  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_bool_float64_64(
    bool* toptr,
    const double* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int32_bool_64(
    int32_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_reduce_sum_int64_bool_64(
    int64_t* toptr,
    const bool* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength);

  EXPORT_SYMBOL ERROR
  awkward_regularize_arrayslice_64(
    int64_t* flatheadptr,
    int64_t lenflathead,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_slicearray_ravel_64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t ndim,
    const int64_t* shape,
    const int64_t* strides);

  EXPORT_SYMBOL ERROR
  awkward_slicemissing_check_same(
    bool* same,
    const int8_t* bytemask,
    const int64_t* missingindex,
    int64_t length);

  EXPORT_SYMBOL ERROR
  awkward_sort_bool(
    bool* toptr,
    const bool* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_int8(
    int8_t* toptr,
    const int8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_int16(
    int16_t* toptr,
    const int16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_int32(
    int32_t* toptr,
    const int32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_int64(
    int64_t* toptr,
    const int64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_uint8(
    uint8_t* toptr,
    const uint8_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_uint16(
    uint16_t* toptr,
    const uint16_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_uint32(
    uint32_t* toptr,
    const uint32_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_uint64(
    uint64_t* toptr,
    const uint64_t* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_float32(
    float* toptr,
    const float* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);
  EXPORT_SYMBOL ERROR
  awkward_sort_float64(
    double* toptr,
    const double* fromptr,
    int64_t length,
    const int64_t* offsets,
    int64_t offsetslength,
    int64_t parentslength,
    bool ascending,
    bool stable);

  EXPORT_SYMBOL ERROR
  awkward_sorting_ranges(
    int64_t* toindex,
    int64_t tolength,
    const int64_t* parents,
    int64_t parentslength);

  EXPORT_SYMBOL ERROR
  awkward_sorting_ranges_length(
    int64_t* tolength,
    const int64_t* parents,
    int64_t parentslength);

  EXPORT_SYMBOL ERROR
  awkward_zero_mask64(
    int64_t* tomask,
    int64_t length);
  EXPORT_SYMBOL ERROR
  awkward_zero_mask8(
    int8_t* tomask,
    int64_t length);

}
#endif
