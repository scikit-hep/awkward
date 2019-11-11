// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_IDENTITY_H_
#define AWKWARDCPU_IDENTITY_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  struct Error awkward_new_identity32(int32_t* toptr, int64_t length);
  struct Error awkward_new_identity64(int64_t* toptr, int64_t length);

  struct Error awkward_identity32_to_identity64(int64_t* toptr, const int32_t* fromptr, int64_t length);

  struct Error awkward_identity32_from_listarray32(int32_t* toptr, const int32_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
  struct Error awkward_identity64_from_listarray32(int64_t* toptr, const int64_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
  struct Error awkward_identity64_from_listarrayU32(int64_t* toptr, const int64_t* fromptr, const uint32_t* fromstarts, const uint32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
  struct Error awkward_identity64_from_listarray64(int64_t* toptr, const int64_t* fromptr, const int64_t* fromstarts, const int64_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
}

#endif // AWKWARDCPU_IDENTITY_H_
