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

  EXPORT_SYMBOL struct Error awkward_indexedarray32_flatten_nextcarry_64(int64_t* tocarry, const int32_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);
  EXPORT_SYMBOL struct Error awkward_indexedarray64_flatten_nextcarry_64(int64_t* tocarry, const int64_t* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent);

  EXPORT_SYMBOL struct Error awkward_listoffsetarray32_flatten_nextcarry(int32_t* tocarry, int64_t* tolen, const int32_t* fromoffsets, int64_t lenoffset, int64_t step);
  EXPORT_SYMBOL struct Error awkward_listoffsetarrayU32_flatten_nextcarry(uint32_t* tocarry, int64_t* tolen, const uint32_t* fromoffsets, int64_t lenoffset, int64_t step);
  EXPORT_SYMBOL struct Error awkward_listoffsetarray64_flatten_nextcarry(int64_t* tocarry, int64_t* tolen, const int64_t* fromoffsets, int64_t lenoffset, int64_t step);
}

#endif // AWKWARDCPU_GETITEM_H_
