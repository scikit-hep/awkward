// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_GETITEM_H_
#define AWKWARDCPU_GETITEM_H_

#include "awkward/cpu-kernels/util.h"

extern "C" {
  void awkward_regularize_rangeslice_64(int64_t& start, int64_t& stop, bool posstep, bool hasstart, bool hasstop, int64_t length);

  Error awkward_getitem();
}

#endif // AWKWARDCPU_GETITEM_H_
