// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H_
#define AWKWARD_KERNEL_UTILS_H_

#include "common.h"
#include <optional>

namespace awkward::kernel {
  /// Normalize a scalar Python index; callers retain their own error context.
  /// As with the kernel specification, length is a nonnegative list size.
  [[nodiscard]] constexpr std::optional<int64_t>
  normalize_index(int64_t index, int64_t length) noexcept {
    if (index < 0) {
      index += length;
    }
    if (!(0 <= index && index < length)) {
      return std::nullopt;
    }
    return index;
  }
}

extern "C" {

  EXPORT_SYMBOL void
    awkward_regularize_rangeslice(
      std::int64_t* start,
      std::int64_t* stop,
      bool posstep,
      bool hasstart,
      bool hasstop,
      std::int64_t length
    );

  EXPORT_SYMBOL void
    awkward_ListArray_combinations_step_64(
      std::int64_t** tocarry,
      std::int64_t* toindex,
      std::int64_t* fromindex,
      std::int64_t j,
      std::int64_t stop,
      std::int64_t n,
      bool replacement
    );

}

#endif // AWKWARD_KERNEL_UTILS_H_
