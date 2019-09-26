// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_IDENTITY_H_
#define AWKWARDCPU_IDENTITY_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  void awkward_new_identity32(int32_t* toptr, int64_t length);
  void awkward_new_identity64(int64_t* toptr, int64_t length);

  void awkward_identity32_to_identity64(int64_t* toptr, const int32_t* fromptr, int64_t length);

  void awkward_identity32_from_listoffsets32(int32_t* toptr, const int32_t* fromptr, int64_t tolength, const int32_t* offsets, int64_t width, int64_t length);
  void awkward_identity64_from_listoffsets32(int64_t* toptr, const int64_t* fromptr, int64_t tolength, const int32_t* offsets, int64_t width, int64_t length);
  void awkward_identity64_from_listoffsets64(int64_t* toptr, const int64_t* fromptr, int64_t tolength, const int64_t* offsets, int64_t width, int64_t length);

  Error awkward_identity32_from_listarray32(int32_t* toptr, const int32_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
  Error awkward_identity64_from_listarray32(int64_t* toptr, const int64_t* fromptr, const int32_t* fromstarts, const int32_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
  Error awkward_identity64_from_listarray64(int64_t* toptr, const int64_t* fromptr, const int64_t* fromstarts, const int64_t* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
}

#endif // AWKWARDCPU_IDENTITY_H_
