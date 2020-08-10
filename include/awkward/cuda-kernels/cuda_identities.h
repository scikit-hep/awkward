// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE


#ifndef AWKWARD_IDENTITIES_H
#define AWKWARD_IDENTITIES_H

#include "awkward/common.h"

extern "C" {
  ERROR awkward_cuda_Identities32_to_Identities64(
  int64_t* toptr,
  const int32_t* fromptr,
  int64_t length,
  int64_t width);
};

#endif //AWKWARD_IDENTITIES_H
