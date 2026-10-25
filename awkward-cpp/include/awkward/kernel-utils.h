// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H_
#define AWKWARD_KERNEL_UTILS_H_

#include "common.h"
#include <limits>
#include <type_traits>

namespace awkward::kernel {
  // Convert a modulo-2**N accumulator without out-of-range signed casts.
  template <typename T>
  T signed_from_unsigned(std::make_unsigned_t<T> value) noexcept {
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
    using U = std::make_unsigned_t<T>;
    if (value <= static_cast<U>(std::numeric_limits<T>::max())) {
      return static_cast<T>(value);
    }
    return T{-1} - static_cast<T>(static_cast<U>(~value));
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
