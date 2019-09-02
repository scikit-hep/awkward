// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_IDENTITY_H_
#define AWKWARDCPU_IDENTITY_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  Error awkward_identity_new32(int64_t length, int32_t* to);
  Error awkward_identity_new64(int64_t length, int32_t* to);
  Error awkward_identity_32to64(int64_t length, int32_t* from, int64_t* to);
  Error awkward_identity_from_listfoffsets32(int64_t length, int64_t width, int32_t* offsets, int32_t* from, int64_t tolength, int32_t* to);
  Error awkward_identity_from_listfoffsets64(int64_t length, int64_t width, int64_t* offsets, int64_t* from, int64_t tolength, int64_t* to);
}

#endif // AWKWARDCPU_IDENTITY_H_
