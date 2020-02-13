// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_REDUCERS_H_
#define AWKWARDCPU_REDUCERS_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  EXPORT_SYMBOL struct Error awkward_content_reduce_zeroparents_64(int64_t* toparents, int64_t length);

}

#endif // AWKWARDCPU_REDUCERS_H_
