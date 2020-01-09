// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_OPERATIONS_H_
#define AWKWARDCPU_OPERATIONS_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error awkward_listarray32_flatten_64(int32_t* tostarts, int32_t* tostops, const int32_t* fromstarts, const int32_t* fromstops, const int64_t lenstarts, int64_t* toindxarray, int64_t* tolen);
  EXPORT_SYMBOL struct Error awkward_listarrayU32_flatten_64(uint32_t* tostarts, uint32_t* tostops, const uint32_t* fromstarts, const uint32_t* fromstops, const int64_t lenstarts, int64_t* toindxarray, int64_t* tolen);
  EXPORT_SYMBOL struct Error awkward_listarray64_flatten_64(int64_t* tostarts, int64_t* tostops, const int64_t* fromstarts, const int64_t* fromstops, const int64_t lenstarts, int64_t* toindxarray, int64_t* tolen);
}

#endif // AWKWARDCPU_GETITEM_H_
