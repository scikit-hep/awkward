// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_NUMERIC_SORT_H_
#define AWKWARD_NUMERIC_SORT_H_

#include <cmath>
#include <type_traits>

namespace awkward::kernel {
  // Shared by value sort and argsort. Integers compare at their native width;
  // NaNs precede all other values in both directions, matching kernel behavior.
  // Equivalent values, including pairs of NaNs and signed zeros, compare false.
  template <typename T, bool Ascending>
  struct NumericOrder {
    [[nodiscard]] bool operator()(T left, T right) const noexcept {
      if constexpr (!std::is_integral_v<T>) {
        if (std::isnan(right)) {
          return false;
        }
        if (std::isnan(left)) {
          return true;
        }
      }
      if constexpr (Ascending) {
        return left < right;
      } else {
        return left > right;
      }
    }
  };
}

#endif // AWKWARD_NUMERIC_SORT_H_
