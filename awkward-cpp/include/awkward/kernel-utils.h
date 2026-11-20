// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_KERNEL_UTILS_H_
#define AWKWARD_KERNEL_UTILS_H_

#include "common.h"
#include <limits>
#include <optional>
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
    if (count == 0) {
      return success();
    }
    if constexpr (std::is_signed_v<Offset>) {
      if (offsets[0] < 0) {
        return failure("offsets must be non-negative", 0, kSliceNone, filename);
      }
    }
    if (static_cast<uint64_t>(offsets[0]) > static_cast<uint64_t>(length)) {
      return failure("offsets exceed length", 0, kSliceNone, filename);
    }
    if (offsets[0] != 0) {
      return failure("offsets[0] must be zero", 0, kSliceNone, filename);
    }
    // A branch-free block scan can vectorize on the valid-input path. Casting
    // negative signed offsets to uint64 also marks them out of range. Replay
    // only the first failing block to preserve position and error precedence.
    for (int64_t begin = 1; begin < count;) {
      const int64_t end = begin + std::min<int64_t>(256, count - begin);
      bool invalid = false;
      for (int64_t i = begin; i < end; ++i) {
        invalid |= (static_cast<uint64_t>(offsets[i]) > static_cast<uint64_t>(length))
                   | (offsets[i] < offsets[i - 1]);
      }
      if (invalid) {
        for (int64_t i = begin; i < end; ++i) {
          if constexpr (std::is_signed_v<Offset>) {
            if (offsets[i] < 0) {
              return failure("offsets must be non-negative", i, kSliceNone, filename);
            }
          }
          if (static_cast<uint64_t>(offsets[i]) > static_cast<uint64_t>(length)) {
            return failure("offsets exceed length", i, kSliceNone, filename);
          }
          if (offsets[i] < offsets[i - 1]) {
            return failure("offsets must be nondecreasing", i, kSliceNone, filename);
          }
        }
      }
      begin = end;
    }
    return success();
  }

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
