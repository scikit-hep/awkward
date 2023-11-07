// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H_
#define AWKWARD_KERNEL_UTILS_H_

#include "common.h"

extern "C" {

  EXPORT_SYMBOL void
    awkward_regularize_rangeslice(
      int64_t* start,
      int64_t* stop,
      bool posstep,
      bool hasstart,
      bool hasstop,
      int64_t length
    );

  EXPORT_SYMBOL void
    awkward_ListArray_combinations_step_64(
      int64_t** tocarry,
      int64_t* toindex,
      int64_t* fromindex,
      int64_t j,
      int64_t stop,
      int64_t n,
      bool replacement
    );

}

#endif // AWKWARD_KERNEL_UTILS_H_
