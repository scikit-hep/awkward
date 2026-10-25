// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H_
#define AWKWARD_KERNEL_UTILS_H_

#include "common.h"
#include <type_traits>

namespace awkward::kernel {
  // Backing allocations must contain the declared element counts. Validate
  // logical offsets before narrowing them or forming any buffer pointers.
  template <typename Offset>
  Error validate_sort_offsets(const Offset* offsets, int64_t count,
                              int64_t length, const char* filename) {
    if (length < 0) {
      return failure("length must be non-negative", kSliceNone, kSliceNone, filename);
    }
    if (count < 0) {
      return failure("offsetslength must be non-negative", kSliceNone, kSliceNone, filename);
    }
    for (int64_t i = 0; i < count; ++i) {
      if constexpr (std::is_signed_v<Offset>) {
        if (offsets[i] < 0) {
          return failure("offsets must be non-negative", i, kSliceNone, filename);
        }
      }
      if (static_cast<uint64_t>(offsets[i]) > static_cast<uint64_t>(length)) {
        return failure("offsets exceed length", i, kSliceNone, filename);
      }
      if (i == 0 && offsets[i] != 0) {
        return failure("offsets[0] must be zero", i, kSliceNone, filename);
      }
      if (i > 0 && offsets[i] < offsets[i - 1]) {
        return failure("offsets must be nondecreasing", i, kSliceNone, filename);
      }
    }
    return success();
  }
}

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
