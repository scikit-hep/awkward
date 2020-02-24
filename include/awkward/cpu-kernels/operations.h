// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_OPERATIONS_H_
#define AWKWARDCPU_OPERATIONS_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error awkward_listarray32_count(int32_t* tocount, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_count(uint32_t* tocount, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_count(int64_t* tocount, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);

  EXPORT_SYMBOL struct Error awkward_listarray32_count_64(int64_t* tocount, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_count_64(int64_t* tocount, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_count_64(int64_t* tocount, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray32_count(int32_t* tocount, const int32_t* fromoffsets, int64_t lenoffsets);
  EXPORT_SYMBOL struct Error awkward_listoffsetarrayU32_count(uint32_t* tocount, const uint32_t* fromoffsets, int64_t lenoffsets);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray64_count(int64_t* tocount, const int64_t* fromoffsets, int64_t lenoffsets);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray32_count_64(int64_t* tocount, const int32_t* fromoffsets, int64_t lenoffsets);
  EXPORT_SYMBOL struct Error awkward_listoffsetarrayU32_count_64(int64_t* tocount, const uint32_t* fromoffsets, int64_t lenoffsets);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray64_count_64(int64_t* tocount, const int64_t* fromoffsets, int64_t lenoffsets);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const int32_t* fromindex, int64_t lenindex, int64_t indexoffset);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const uint32_t* fromindex, int64_t lenindex, int64_t indexoffset);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_count(int64_t* tocount, const int64_t* contentcount, int64_t lencontent, const int64_t* fromindex, int64_t lenindex, int64_t indexoffset);

  EXPORT_SYMBOL struct Error awkward_regulararray_count(int64_t* tocount, int64_t size, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listarray32_flatten_length(int64_t* tolen, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_flatten_length(int64_t* tolen, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_flatten_length(int64_t* tolen, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray32_flatten_64(int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_flatten_64(int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_flatten_64(int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);

  EXPORT_SYMBOL struct Error awkward_listarray32_flatten_scale_64(int32_t* tostarts, int32_t* tostops, const int64_t* scale, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_flatten_scale_64(uint32_t* tostarts, uint32_t* tostops, const int64_t* scale, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_flatten_scale_64(int64_t* tostarts, int64_t* tostops, const int64_t* scale, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_flatten_nextcarry_64(int64_t* tocarry, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_flatten_nextcarry_64(int64_t* tocarry, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_flatten_nextcarry_64(int64_t* tocarry, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const int32_t* fromindex, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const uint32_t* fromindex, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_overlay_mask8_to64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const int64_t* fromindex, int64_t indexoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_mask8(int8_t* tomask, const int32_t* fromindex, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_mask8(int8_t* tomask, const uint32_t* fromindex, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_mask8(int8_t* tomask, const int64_t* fromindex, int64_t indexoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_zero_mask8(int8_t* tomask, int64_t length);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_simplify32_to64(int64_t* toindex, const int32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray32_simplifyU32_to64(int64_t* toindex, const int32_t* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray32_simplify64_to64(int64_t* toindex, const int32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_simplify32_to64(int64_t* toindex, const uint32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_simplifyU32_to64(int64_t* toindex, const uint32_t* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_simplify64_to64(int64_t* toindex, const uint32_t* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_simplify32_to64(int64_t* toindex, const int64_t* outerindex, int64_t outeroffset, int64_t outerlength, const int32_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_simplifyU32_to64(int64_t* toindex, const int64_t* outerindex, int64_t outeroffset, int64_t outerlength, const uint32_t* innerindex, int64_t inneroffset, int64_t innerlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_simplify64_to64(int64_t* toindex, const int64_t* outerindex, int64_t outeroffset, int64_t outerlength, const int64_t* innerindex, int64_t inneroffset, int64_t innerlength);

  EXPORT_SYMBOL struct Error awkward_regulararray_compact_offsets64(int64_t* tooffsets, int64_t length, int64_t size);

  EXPORT_SYMBOL struct Error awkward_listarray32_compact_offsets64(int64_t* tooffsets, const int32_t* fromstarts, const int32_t* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_compact_offsets64(int64_t* tooffsets, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listarray64_compact_offsets64(int64_t* tooffsets, const int64_t* fromstarts, const int64_t* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray32_compact_offsets64(int64_t* tooffsets, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarrayU32_compact_offsets64(int64_t* tooffsets, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray64_compact_offsets64(int64_t* tooffsets, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listarray32_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const int32_t* fromstarts, int64_t startsoffset, const int32_t* fromstops, int64_t stopsoffset, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const uint32_t* fromstarts, int64_t startsoffset, const uint32_t* fromstops, int64_t stopsoffset, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_listarray64_broadcast_tooffsets64(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, const int64_t* fromstarts, int64_t startsoffset, const int64_t* fromstops, int64_t stopsoffset, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_regulararray_broadcast_tooffsets64(const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength, int64_t size);

  EXPORT_SYMBOL struct Error awkward_regulararray_broadcast_tooffsets64_size1(int64_t* tocarry, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray32_toRegularArray(int64_t* size, const int32_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength);
  EXPORT_SYMBOL struct Error awkward_listoffsetarrayU32_toRegularArray(int64_t* size, const uint32_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray64_toRegularArray(int64_t* size, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t offsetslength);

  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_fromdouble(double* toptr, int64_t tooffset, const double* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_fromfloat(double* toptr, int64_t tooffset, const float* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_from64(double* toptr, int64_t tooffset, const int64_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_fromU64(double* toptr, int64_t tooffset, const uint64_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_from32(double* toptr, int64_t tooffset, const int32_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_fromU32(double* toptr, int64_t tooffset, const uint32_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_from16(double* toptr, int64_t tooffset, const int16_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_fromU16(double* toptr, int64_t tooffset, const uint16_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_from8(double* toptr, int64_t tooffset, const int8_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_fromU8(double* toptr, int64_t tooffset, const uint8_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_todouble_frombool(double* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_toU64_fromU64(uint64_t* toptr, int64_t tooffset, const uint64_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_from64(int64_t* toptr, int64_t tooffset, const int64_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_fromU64(int64_t* toptr, int64_t tooffset, const uint64_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_from32(int64_t* toptr, int64_t tooffset, const int32_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_fromU32(int64_t* toptr, int64_t tooffset, const uint32_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_from16(int64_t* toptr, int64_t tooffset, const int16_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_fromU16(int64_t* toptr, int64_t tooffset, const uint16_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_from8(int64_t* toptr, int64_t tooffset, const int8_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_fromU8(int64_t* toptr, int64_t tooffset, const uint8_t* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_to64_frombool(int64_t* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_numpyarray_fill_tobool_frombool(bool* toptr, int64_t tooffset, const bool* fromptr, int64_t fromoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_zero_index_64(int64_t* toindex, int64_t length);
  EXPORT_SYMBOL struct Error awkward_zero_raw_ptr(uint8_t* toptr, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index_pad(int64_t* toindex, const int64_t fromlength, int64_t tolength);
  EXPORT_SYMBOL struct Error awkward_index_inject_pad(int64_t* toindex, const int64_t* fromindex, int64_t shape, int64_t chunks, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index_clip(int64_t* toindex, const int64_t* fromindex, int64_t tolength);
  EXPORT_SYMBOL struct Error awkward_indexedarray_pad_to64_from32(int64_t* toindex, const int32_t* fromindex, int64_t tolength, int64_t fromlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray_pad_to64_fromU32(int64_t* toindex, const uint32_t* fromindex, int64_t tolength, int64_t fromlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray_pad_to64_from64(int64_t* toindex, const int64_t* fromindex, int64_t tolength, int64_t fromlength);
  EXPORT_SYMBOL struct Error awkward_indexedarray_inject_pad_from32(int64_t* toindex, const int32_t* fromindex, int64_t tolength, int64_t fromlength, int64_t fromsize);
  EXPORT_SYMBOL struct Error awkward_indexedarray_inject_pad_fromU32(int64_t* toindex, const uint32_t* fromindex, int64_t tolength, int64_t fromlength, int64_t fromsize);
  EXPORT_SYMBOL struct Error awkward_indexedarray_inject_pad_from64(int64_t* toindex, const int64_t* fromindex, int64_t tolength, int64_t fromlength, int64_t fromsize);
  EXPORT_SYMBOL struct Error awkward_indexedarray_clip(int64_t* toindex, int64_t* fromindex, int64_t tolength);

  EXPORT_SYMBOL struct Error awkward_regulararray_pad(int64_t* toindex, int64_t tolength, int64_t fromlength);

  EXPORT_SYMBOL struct Error awkward_numpyarray_pad_copy_64(uint8_t* toptr, const uint8_t* fromptr, int64_t tolen, int64_t fromlen, int64_t tostride, int64_t fromstride, int64_t offset, const int64_t* pos);

  EXPORT_SYMBOL struct Error awkward_listarray_fill_to64_from32(int64_t* tostarts, int64_t tostartsoffset, int64_t* tostops, int64_t tostopsoffset, const int32_t* fromstarts, int64_t fromstartsoffset, const int32_t* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_listarray_fill_to64_fromU32(int64_t* tostarts, int64_t tostartsoffset, int64_t* tostops, int64_t tostopsoffset, const uint32_t* fromstarts, int64_t fromstartsoffset, const uint32_t* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_listarray_fill_to64_from64(int64_t* tostarts, int64_t tostartsoffset, int64_t* tostops, int64_t tostopsoffset, const int64_t* fromstarts, int64_t fromstartsoffset, const int64_t* fromstops, int64_t fromstopsoffset, int64_t length, int64_t base);

  EXPORT_SYMBOL struct Error awkward_indexedarray_fill_to64_from32(int64_t* toindex, int64_t toindexoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_indexedarray_fill_to64_fromU32(int64_t* toindex, int64_t toindexoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_indexedarray_fill_to64_from64(int64_t* toindex, int64_t toindexoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t length, int64_t base);

  EXPORT_SYMBOL struct Error awkward_indexedarray_fill_to64_count(int64_t* toindex, int64_t toindexoffset, int64_t length, int64_t base);

  EXPORT_SYMBOL struct Error awkward_unionarray_filltags_to8_from8(int8_t* totags, int64_t totagsoffset, const int8_t* fromtags, int64_t fromtagsoffset, int64_t length, int64_t base);

  EXPORT_SYMBOL struct Error awkward_unionarray_fillindex_to64_from32(int64_t* toindex, int64_t toindexoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_unionarray_fillindex_to64_fromU32(int64_t* toindex, int64_t toindexoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_unionarray_fillindex_to64_from64(int64_t* toindex, int64_t toindexoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_unionarray_filltags_to8_const(int8_t* totags, int64_t totagsoffset, int64_t length, int64_t base);

  EXPORT_SYMBOL struct Error awkward_unionarray_fillindex_to64_count(int64_t* toindex, int64_t toindexoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_unionarray8_32_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_32_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_32_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_U32_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const uint32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_U32_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const uint32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_U32_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const uint32_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_64_simplify8_32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int64_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_64_simplify8_U32_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int64_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const uint32_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_64_simplify8_64_to8_64(int8_t* totags, int64_t* toindex, const int8_t* outertags, int64_t outertagsoffset, const int64_t* outerindex, int64_t outerindexoffset, const int8_t* innertags, int64_t innertagsoffset, const int64_t* innerindex, int64_t innerindexoffset, int64_t towhich, int64_t innerwhich, int64_t outerwhich, int64_t length, int64_t base);

  EXPORT_SYMBOL struct Error awkward_unionarray8_32_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const int8_t* fromtags, int64_t fromtagsoffset, const int32_t* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_U32_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const int8_t* fromtags, int64_t fromtagsoffset, const uint32_t* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base);
  EXPORT_SYMBOL struct Error awkward_unionarray8_64_simplify_one_to8_64(int8_t* totags, int64_t* toindex, const int8_t* fromtags, int64_t fromtagsoffset, const int64_t* fromindex, int64_t fromindexoffset, int64_t towhich, int64_t fromwhich, int64_t length, int64_t base);

}

#endif // AWKWARDCPU_GETITEM_H_
