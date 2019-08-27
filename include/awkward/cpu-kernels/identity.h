// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_IDENTITY_H_
#define AWKWARDCPU_IDENTITY_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  Error awkward_identity_numpyarray_newid(IndexType* ptr, IndexType length);
}

#endif // AWKWARDCPU_IDENTITY_H_
