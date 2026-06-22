// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort.cpp", line)

#include "awkward/kernels.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

// Per-segment sort by value. Integer types compare directly (no isnan, no
// conversion to double — casting int64/uint64 through double would round
// values beyond 2^53 and break strict weak ordering). Floating-point types
// keep the NaN-aware path: NaNs compare "less" than everything, regardless
// of direction, so they are pushed to the low end (same behavior as the
// older hand-rolled comparator; note this differs from NumPy, which sorts
// NaNs to the high end).
//
// Explicit specializations must appear before implicit instantiations.
// sort_order_ascending/descending for bool: no NaN, just direct comparison.
template <typename T>
bool sort_order_ascending(T l, T r);

template <typename T>
bool sort_order_descending(T l, T r);

template <>
bool sort_order_ascending(bool l, bool r)
{
  return l < r;
}

template <>
bool sort_order_descending(bool l, bool r)
{
  return l > r;
}

template <typename T>
bool sort_order_ascending(T l, T r)
{
  if constexpr (std::is_integral_v<T>) {
    return l < r;
  } else {
    return !std::isnan(r) && (std::isnan(l) || l < r);
  }
}

template <typename T>
bool sort_order_descending(T l, T r)
{
  if constexpr (std::is_integral_v<T>) {
    return l > r;
  } else {
    return !std::isnan(r) && (std::isnan(l) || l > r);
  }
}

template <typename T>
ERROR awkward_sort(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), int64_t{0});

  // Each segment of `index` is sorted independently.
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    // Clamp against `length` so malformed offsets cannot index past the end
    // of `index` (well-formed offsets satisfy offsets[last] <= length).
    int64_t start_off = (offsets[i] < length) ? offsets[i] : length;
    int64_t stop_off = (offsets[i + 1] < length) ? offsets[i + 1] : length;
    if (start_off >= stop_off) continue;
    auto start = index.begin() + start_off;
    auto stop = index.begin() + stop_off;
    // Hoist the ascending/stable dispatch outside the per-element
    // comparisons (one branch per segment instead of O(n log n)).
    if (stable) {
      if (ascending) {
        std::stable_sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return sort_order_ascending<T>(fromptr[i1], fromptr[i2]);
        });
      } else {
        std::stable_sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return sort_order_descending<T>(fromptr[i1], fromptr[i2]);
        });
      }
    } else {
      if (ascending) {
        std::sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return sort_order_ascending<T>(fromptr[i1], fromptr[i2]);
        });
      } else {
        std::sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return sort_order_descending<T>(fromptr[i1], fromptr[i2]);
        });
      }
    }
  }

  // In the offsets representation, the number of elements to copy out is
  // simply the end of the last bin (offsets[0] == 0 by contract).
  int64_t total = (offsetslength > 0) ? offsets[offsetslength - 1] : 0;
  int64_t copy_length = (total < length) ? total : length;
  for (int64_t i = 0; i < copy_length; i++) {
    toptr[i] = fromptr[index[i]];
  }
  return success();
}

#define SORT(T, NAME)                                                      \
  ERROR awkward_sort_##NAME(                                               \
    T* toptr, const T* fromptr, int64_t length,                            \
    const int64_t* offsets, int64_t offsetslength,                         \
    bool ascending, bool stable) {                                         \
    return awkward_sort<T>(                                                \
      toptr, fromptr, length, offsets, offsetslength,                      \
      ascending, stable);                                                  \
  }

SORT(bool,     bool)
SORT(int8_t,   int8)
SORT(uint8_t,  uint8)
SORT(int16_t,  int16)
SORT(uint16_t, uint16)
SORT(int32_t,  int32)
SORT(uint32_t, uint32)
SORT(int64_t,  int64)
SORT(uint64_t, uint64)
SORT(float,    float32)
SORT(double,   float64)
