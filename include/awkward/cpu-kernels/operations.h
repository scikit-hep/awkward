// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_OPERATIONS_H_
#define AWKWARDCPU_OPERATIONS_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error awkward_index_to64_from8(int64_t* toindex, const int8_t* fromindex, int64_t length, int64_t offset);
  EXPORT_SYMBOL struct Error awkward_index_to64_fromU8(int64_t* toindex, const uint8_t* fromindex, int64_t length, int64_t offset);
  EXPORT_SYMBOL struct Error awkward_index_to64_from16(int64_t* toindex, const int16_t* fromindex, int64_t length, int64_t offset);
  EXPORT_SYMBOL struct Error awkward_index_to64_fromU16(int64_t* toindex, const uint16_t* fromindex, int64_t length, int64_t offset);
  EXPORT_SYMBOL struct Error awkward_index_to64_from32(int64_t* toindex, const int32_t* fromindex, int64_t length, int64_t offset);
  EXPORT_SYMBOL struct Error awkward_index_to64_fromU32(int64_t* toindex, const uint32_t* fromindex, int64_t length, int64_t offset);
  EXPORT_SYMBOL struct Error awkward_index_to64_from64(int64_t* toindex, const int64_t* fromindex, int64_t length, int64_t offset);

  EXPORT_SYMBOL struct Error awkward_listarray32_count(int32_t* tocount, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_count(uint32_t* tocount, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_count(int64_t* tocount, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);

  EXPORT_SYMBOL struct Error awkward_regulararray_count(int64_t* tocount, int64_t size, int64_t length);

  EXPORT_SYMBOL struct Error awkward_listarray32_flatten_length(int64_t* tolen, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_flatten_length(int64_t* tolen, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_flatten_length(int64_t* tolen, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray32_flatten_64(int64_t* tocarry, const int32_t* fromstarts, const int32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_flatten_64(int64_t* tocarry, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
  EXPORT_SYMBOL struct Error awkward_listarray64_flatten_64(int64_t* tocarry, const int64_t* fromstarts, const int64_t* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset);
}

#endif // AWKWARDCPU_GETITEM_H_
