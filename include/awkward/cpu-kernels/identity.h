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
}

#endif // AWKWARDCPU_IDENTITY_H_
