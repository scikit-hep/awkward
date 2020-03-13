// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_GETITEM_H_
#define AWKWARDCPU_GETITEM_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL void awkward_regularize_rangeslice(int64_t* start, int64_t* stop, bool posstep, bool hasstart, bool hasstop, int64_t length);
  EXPORT_SYMBOL struct Error awkward_regularize_arrayslice_64(int64_t* flatheadptr, int64_t lenflathead, int64_t length);

  EXPORT_SYMBOL struct Error awkward_index8_to_index64(int64_t* toptr, const int8_t* fromptr, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexU8_to_index64(int64_t* toptr, const uint8_t* fromptr, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index32_to_index64(int64_t* toptr, const int32_t* fromptr, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexU32_to_index64(int64_t* toptr, const uint32_t* fromptr, int64_t length);

  EXPORT_SYMBOL struct Error awkward_index8_carry_64(int8_t* toindex, const int8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexU8_carry_64(uint8_t* toindex, const uint8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index32_carry_64(int32_t* toindex, const int32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexU32_carry_64(uint32_t* toindex, const uint32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index64_carry_64(int64_t* toindex, const int64_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t lenfromindex, int64_t length);

  EXPORT_SYMBOL struct Error awkward_index8_carry_nocheck_64(int8_t* toindex, const int8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexU8_carry_nocheck_64(uint8_t* toindex, const uint8_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index32_carry_nocheck_64(int32_t* toindex, const int32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexU32_carry_nocheck_64(uint32_t* toindex, const uint32_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_index64_carry_nocheck_64(int64_t* toindex, const int64_t* fromindex, const int64_t* carry, int64_t fromindexoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_slicearray_ravel_64(int64_t* toptr, const int64_t* fromptr, int64_t ndim, const int64_t* shape, const int64_t* strides);

  EXPORT_SYMBOL struct Error awkward_slicejagged_tocarrylen_tooffsets_64(int64_t* tocarrylen, int64_t* tooffsets, const int64_t* fromoffsets, int64_t fromoffsetsoffset, int64_t fromoffsetslen, const int64_t* fromcarry, int64_t fromcarryoffset, int64_t fromcarrylen);
  EXPORT_SYMBOL struct Error awkward_slicejagged_tocarry_64(int64_t* tocarry, const int64_t* fromoffsets, int64_t fromoffsetsoffset, int64_t fromoffsetslen, const int64_t* fromcarry, int64_t fromcarryoffset, int64_t fromcarrylen);

  EXPORT_SYMBOL struct Error awkward_slicemasked_project_numnull_64(int64_t* numnull, const int64_t* index, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_slicemasked_project_nextcarry_64(int64_t* tocarry, const int64_t* index, int64_t indexoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_carry_arange_64(int64_t* toptr, int64_t length);

  EXPORT_SYMBOL struct Error awkward_identities32_getitem_carry_64(int32_t* newidentitiesptr, const int32_t* identitiesptr, const int64_t* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length);
  EXPORT_SYMBOL struct Error awkward_identities64_getitem_carry_64(int64_t* newidentitiesptr, const int64_t* identitiesptr, const int64_t* carryptr, int64_t lencarry, int64_t offset, int64_t width, int64_t length);

  EXPORT_SYMBOL struct Error awkward_numpyarray_contiguous_init_64(int64_t* toptr, int64_t skip, int64_t stride);
  EXPORT_SYMBOL struct Error awkward_numpyarray_contiguous_copy_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos);
  EXPORT_SYMBOL struct Error awkward_numpyarray_contiguous_next_64(int64_t* topos, const int64_t* frompos, int64_t len, int64_t skip, int64_t stride);

  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_next_null_64(uint8_t* toptr, const uint8_t* fromptr, int64_t len, int64_t stride, int64_t offset, const int64_t* pos);
  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_next_at_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t skip, int64_t at);
  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_next_range_64(int64_t* nextcarryptr, const int64_t* carryptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step);
  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_next_range_advanced_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* advancedptr, int64_t lencarry, int64_t lenhead, int64_t skip, int64_t start, int64_t step);
  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_next_array_64(int64_t* nextcarryptr, int64_t* nextadvancedptr, const int64_t* carryptr, const int64_t* flatheadptr, int64_t lencarry, int64_t lenflathead, int64_t skip);
  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_next_array_advanced_64(int64_t* nextcarryptr, const int64_t* carryptr, const int64_t* advancedptr, const int64_t* flatheadptr, int64_t lencarry, int64_t skip);

  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_boolean_numtrue(int64_t* numtrue, const int8_t* fromptr, int64_t byteoffset, int64_t length, int64_t stride);
  EXPORT_SYMBOL struct Error awkward_numpyarray_getitem_boolean_nonzero_64(int64_t* toptr, const int8_t* fromptr, int64_t byteoffset, int64_t length, int64_t stride);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_at_64(int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_at_64(int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_at_64(int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_range_carrylength(int64_t* carrylength, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_range_carrylength(int64_t* carrylength, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_range_carrylength(int64_t* carrylength, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_range_64(int32_t* tooffsets, int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_range_64(uint32_t* tooffsets, int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_range_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_range_counts_64(int64_t* total, const int32_t* fromoffsets, int64_t lenstarts);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_range_counts_64(int64_t* total, const uint32_t* fromoffsets, int64_t lenstarts);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_range_counts_64(int64_t* total, const int64_t* fromoffsets, int64_t lenstarts);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const int32_t* fromoffsets, int64_t lenstarts);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const uint32_t* fromoffsets, int64_t lenstarts);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const int64_t* fromoffsets, int64_t lenstarts);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_carry_64(int32_t* tostarts, int32_t* tostops, const int32_t* fromstarts, const int32_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_carry_64(uint32_t* tostarts, uint32_t* tostops, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_carry_64(int64_t* tostarts, int64_t* tostops, const int64_t* fromstarts, const int64_t* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry);

  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_next_at_64(int64_t* tocarry, int64_t at, int64_t len, int64_t size);
  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_next_range_64(int64_t* tocarry, int64_t regular_start, int64_t step, int64_t len, int64_t size, int64_t nextsize);
  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, int64_t len, int64_t nextsize);
  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_next_array_regularize_64(int64_t* toarray, const int64_t* fromarray, int64_t lenarray, int64_t size);
  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromarray, int64_t len, int64_t lenarray, int64_t size);
  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const int64_t* fromadvanced, const int64_t* fromarray, int64_t len, int64_t lenarray, int64_t size);
  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_carry_64(int64_t* tocarry, const int64_t* fromcarry, int64_t lencarry, int64_t size);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_numnull(int64_t* numnull, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_numnull(int64_t* numnull, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_numnull(int64_t* numnull, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_getitem_nextcarry_outindex_64(int64_t* tocarry, int32_t* toindex, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_getitem_nextcarry_outindex_64(int64_t* tocarry, uint32_t* toindex, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_getitem_nextcarry_outindex_64(int64_t* tocarry, int64_t* toindex, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_getitem_nextcarry_outindex_mask_64(int64_t* tocarry, int64_t* toindex, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_getitem_nextcarry_outindex_mask_64(int64_t* tocarry, int64_t* toindex, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_getitem_nextcarry_outindex_mask_64(int64_t* tocarry, int64_t* toindex, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray_getitem_adjust_offsets_64(int64_t* tooffsets, int64_t* tononzero, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t length, const int64_t* nonzero, int64_t nonzerooffset, int64_t nonzerolength);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray_getitem_adjust_offsets_index_64(int64_t* tooffsets, int64_t* tononzero, const int64_t* fromoffsets, int64_t offsetsoffset, int64_t length, const int64_t* index, int64_t indexoffset, int64_t indexlength, const int64_t* nonzero, int64_t nonzerooffset, int64_t nonzerolength, const int8_t* originalmask, int64_t maskoffset, int64_t masklength);

  EXPORT_SYMBOL struct Error awkward_indexedarray_getitem_adjust_outindex_64(int8_t* tomask, int64_t* toindex, int64_t* tononzero, const int64_t* fromindex, int64_t fromindexoffset, int64_t fromindexlength, const int64_t* nonzero, int64_t nonzerooffset, int64_t nonzerolength);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_getitem_nextcarry_64(int64_t* tocarry, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_getitem_nextcarry_64(int64_t* tocarry, const uint32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_getitem_nextcarry_64(int64_t* tocarry, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_indexedarray32_getitem_carry_64(int32_t* toindex, const int32_t* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_getitem_carry_64(uint32_t* toindex, const uint32_t* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_getitem_carry_64(int64_t* toindex, const int64_t* fromindex, const int64_t* fromcarry, int64_t indexoffset, int64_t lenindex, int64_t lencarry);

  EXPORT_SYMBOL struct Error awkward_unionarray8_32_regular_index(int32_t* toindex, const int8_t* fromtags, int64_t tagsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_unionarray8_U32_regular_index(uint32_t* toindex, const int8_t* fromtags, int64_t tagsoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_unionarray8_64_regular_index(int64_t* toindex, const int8_t* fromtags, int64_t tagsoffset, int64_t length);

  EXPORT_SYMBOL struct Error awkward_unionarray8_32_project_64(int64_t* lenout, int64_t* tocarry, const int8_t* fromtags, int64_t tagsoffset, const int32_t* fromindex, int64_t indexoffset, int64_t length, int64_t which);
  EXPORT_SYMBOL struct Error awkward_unionarray8_U32_project_64(int64_t* lenout, int64_t* tocarry, const int8_t* fromtags, int64_t tagsoffset, const uint32_t* fromindex, int64_t indexoffset, int64_t length, int64_t which);
  EXPORT_SYMBOL struct Error awkward_unionarray8_64_project_64(int64_t* lenout, int64_t* tocarry, const int8_t* fromtags, int64_t tagsoffset, const int64_t* fromindex, int64_t indexoffset, int64_t length, int64_t which);

  EXPORT_SYMBOL struct Error awkward_missing_repeat_64(int64_t* outindex, const int64_t* index, int64_t indexoffset, int64_t indexlength, int64_t repetitions, int64_t regularsize);

  EXPORT_SYMBOL struct Error awkward_regulararray_getitem_jagged_expand_64(int64_t* multistarts, int64_t* multistops, const int64_t* singleoffsets, int64_t regularsize, int64_t regularlength);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_jagged_expand_64(int64_t* multistarts, int64_t* multistops, const int64_t* singleoffsets, int64_t* tocarry, const int32_t* fromstarts, int64_t fromstartsoffset, const int32_t* fromstops, int64_t fromstopsoffset, int64_t jaggedsize, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_jagged_expand_64(int64_t* multistarts, int64_t* multistops, const int64_t* singleoffsets, int64_t* tocarry, const uint32_t* fromstarts, int64_t fromstartsoffset, const uint32_t* fromstops, int64_t fromstopsoffset, int64_t jaggedsize, int64_t length);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_jagged_expand_64(int64_t* multistarts, int64_t* multistops, const int64_t* singleoffsets, int64_t* tocarry, const int64_t* fromstarts, int64_t fromstartsoffset, const int64_t* fromstops, int64_t fromstopsoffset, int64_t jaggedsize, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listarray_getitem_jagged_carrylen_64(int64_t* carrylen, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_jagged_apply_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const int64_t* sliceindex, int64_t sliceindexoffset, int64_t sliceinnerlen, const int32_t* fromstarts, int64_t fromstartsoffset, const int32_t* fromstops, int64_t fromstopsoffset, int64_t contentlen);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_jagged_apply_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const int64_t* sliceindex, int64_t sliceindexoffset, int64_t sliceinnerlen, const uint32_t* fromstarts, int64_t fromstartsoffset, const uint32_t* fromstops, int64_t fromstopsoffset, int64_t contentlen);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_jagged_apply_64(int64_t* tooffsets, int64_t* tocarry, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const int64_t* sliceindex, int64_t sliceindexoffset, int64_t sliceinnerlen, const int64_t* fromstarts, int64_t fromstartsoffset, const int64_t* fromstops, int64_t fromstopsoffset, int64_t contentlen);

  EXPORT_SYMBOL struct Error awkward_listarray_getitem_jagged_numvalid_64(int64_t* numvalid, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t length, const int64_t* missing, int64_t missingoffset, int64_t missinglength);

  EXPORT_SYMBOL struct Error awkward_listarray_getitem_jagged_shrink_64(int64_t* tocarry, int64_t* tosmalloffsets, int64_t* tolargeoffsets, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t length, const int64_t* missing, int64_t missingoffset);

  EXPORT_SYMBOL struct Error awkward_listarray32_getitem_jagged_descend_64(int64_t* tooffsets, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const int32_t* fromstarts, int64_t fromstartsoffset, const int32_t* fromstops, int64_t fromstopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_getitem_jagged_descend_64(int64_t* tooffsets, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const uint32_t* fromstarts, int64_t fromstartsoffset, const uint32_t* fromstops, int64_t fromstopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_getitem_jagged_descend_64(int64_t* tooffsets, const int64_t* slicestarts, int64_t slicestartsoffset, const int64_t* slicestops, int64_t slicestopsoffset, int64_t sliceouterlen, const int64_t* fromstarts, int64_t fromstartsoffset, const int64_t* fromstops, int64_t fromstopsoffset);

  EXPORT_SYMBOL struct Error awkward_bytemaskedarray_getitem_carry_64(int8_t* tomask, const int8_t* frommask, int64_t frommaskoffset, int64_t lenmask, const int64_t* fromcarry, int64_t lencarry);

  EXPORT_SYMBOL struct Error awkward_bytemaskedarray_numnull(int64_t* numnull, const int8_t* mask, int64_t maskoffset, int64_t length, bool validwhen);
  EXPORT_SYMBOL struct Error awkward_bytemaskedarray_getitem_nextcarry_64(int64_t* tocarry, const int8_t* mask, int64_t maskoffset, int64_t length, bool validwhen);
  EXPORT_SYMBOL struct Error awkward_bytemaskedarray_getitem_nextcarry_outindex_64(int64_t* tocarry, int64_t* outindex, const int8_t* mask, int64_t maskoffset, int64_t length, bool validwhen);

  EXPORT_SYMBOL struct Error awkward_bytemaskedarray_toindexedarray_64(int64_t* toindex, const int8_t* mask, int64_t maskoffset, int64_t length, bool validwhen);

}

#endif // AWKWARDCPU_GETITEM_H_
