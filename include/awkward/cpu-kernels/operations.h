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

  EXPORT_SYMBOL struct Error awkward_indexedarray32_andmask_8(int32_t* toindex, const int8_t* mask, int64_t maskoffset, const int32_t* fromindex, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarrayU32_andmask_8(uint32_t* toindex, const int8_t* mask, int64_t maskoffset, const uint32_t* fromindex, int64_t indexoffset, int64_t length);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_andmask_8(int64_t* toindex, const int8_t* mask, int64_t maskoffset, const int64_t* fromindex, int64_t indexoffset, int64_t length);

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

}

#endif // AWKWARDCPU_GETITEM_H_
