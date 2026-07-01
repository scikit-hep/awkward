// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_argsort.cpp", line)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include "awkward/kernels.h"

// Per-segment argsort. Integer types compare directly (no isnan, no
// conversion to double — casting int64/uint64 through double would round
// values beyond 2^53 and break strict weak ordering). Floating-point types
// keep the NaN-aware path: NaNs compare "less" than everything, regardless
// of direction, so they are pushed to the low end (same behavior as the
// older hand-rolled comparator; note this differs from NumPy, which sorts
// NaNs to the high end).
//
// Explicit specializations must appear before implicit instantiations.
// argsort_order_ascending/descending for bool: no NaN, just direct comparison.
template <typename T>
bool argsort_order_ascending(T l, T r);

template <typename T>
bool argsort_order_descending(T l, T r);

template <>
bool argsort_order_ascending(bool l, bool r)
{
  return l < r;
}

template <>
bool argsort_order_descending(bool l, bool r)
{
  return l > r;
}

template <typename T>
bool argsort_order_ascending(T l, T r)
{
  if constexpr (std::is_integral_v<T>) {
    return l < r;
  } else {
    return !std::isnan(r) && (std::isnan(l) || l < r);
  }
}

template <typename T>
bool argsort_order_descending(T l, T r)
{
  if constexpr (std::is_integral_v<T>) {
    return l > r;
  } else {
    return !std::isnan(r) && (std::isnan(l) || l > r);
  }
}

template <typename T, typename U>
ERROR awkward_argsort(
  int64_t* __restrict__ toptr,
  const T* __restrict__ fromptr,
  int64_t length,
  const U* __restrict__ offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::iota(toptr, toptr + length, int64_t{0});

  // Each segment of `toptr` is sorted independently.
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    // Clamp against `length` so malformed offsets cannot index past the end
    // of `toptr` (well-formed offsets satisfy offsets[last] <= length).
    int64_t start_off = static_cast<int64_t>(offsets[i]);
    int64_t stop_off = static_cast<int64_t>(offsets[i + 1]);
    if (start_off > length) start_off = length;
    if (stop_off > length) stop_off = length;
    if (start_off >= stop_off) continue;
    int64_t* segment_start = toptr + start_off;
    int64_t* segment_stop = toptr + stop_off;
    // Hoist the ascending/stable dispatch outside the per-element
    // comparisons (one branch per segment instead of O(n log n)).
    if (stable) {
      if (ascending) {
        std::stable_sort(segment_start, segment_stop, [fromptr](int64_t i1, int64_t i2) {
          return argsort_order_ascending<T>(fromptr[i1], fromptr[i2]);
        });
      } else {
        std::stable_sort(segment_start, segment_stop, [fromptr](int64_t i1, int64_t i2) {
          return argsort_order_descending<T>(fromptr[i1], fromptr[i2]);
        });
      }
    } else {
      if (ascending) {
        std::sort(segment_start, segment_stop, [fromptr](int64_t i1, int64_t i2) {
          return argsort_order_ascending<T>(fromptr[i1], fromptr[i2]);
        });
      } else {
        std::sort(segment_start, segment_stop, [fromptr](int64_t i1, int64_t i2) {
          return argsort_order_descending<T>(fromptr[i1], fromptr[i2]);
        });
      }
    }
    std::transform(segment_start, segment_stop, segment_start,
                   [start_off](int64_t j) { return j - start_off; });
  }
  return success();
}

#define ARGSORT_OFF(FUNC, T, U)                                            \
  ERROR FUNC(                                                              \
    int64_t* toptr, const T* fromptr, int64_t length,                      \
    const U* offsets, int64_t offsetslength,                               \
    bool ascending, bool stable) {                                         \
    return awkward_argsort<T, U>(                                          \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);  \
  }

ARGSORT_OFF(awkward_argsort_bool_int32, bool, int32_t)
ARGSORT_OFF(awkward_argsort_bool_uint32, bool, uint32_t)
ARGSORT_OFF(awkward_argsort_bool_uint64, bool, uint64_t)
ARGSORT_OFF(awkward_argsort_bool_int64, bool, int64_t)

ARGSORT_OFF(awkward_argsort_int8_int32, int8_t, int32_t)
ARGSORT_OFF(awkward_argsort_int8_uint32, int8_t, uint32_t)
ARGSORT_OFF(awkward_argsort_int8_uint64, int8_t, uint64_t)
ARGSORT_OFF(awkward_argsort_int8_int64, int8_t, int64_t)

ARGSORT_OFF(awkward_argsort_uint8_int32, uint8_t, int32_t)
ARGSORT_OFF(awkward_argsort_uint8_uint32, uint8_t, uint32_t)
ARGSORT_OFF(awkward_argsort_uint8_uint64, uint8_t, uint64_t)
ARGSORT_OFF(awkward_argsort_uint8_int64, uint8_t, int64_t)

ARGSORT_OFF(awkward_argsort_int16_int32, int16_t, int32_t)
ARGSORT_OFF(awkward_argsort_int16_uint32, int16_t, uint32_t)
ARGSORT_OFF(awkward_argsort_int16_uint64, int16_t, uint64_t)
ARGSORT_OFF(awkward_argsort_int16_int64, int16_t, int64_t)

ARGSORT_OFF(awkward_argsort_uint16_int32, uint16_t, int32_t)
ARGSORT_OFF(awkward_argsort_uint16_uint32, uint16_t, uint32_t)
ARGSORT_OFF(awkward_argsort_uint16_uint64, uint16_t, uint64_t)
ARGSORT_OFF(awkward_argsort_uint16_int64, uint16_t, int64_t)

ARGSORT_OFF(awkward_argsort_int32_int32, int32_t, int32_t)
ARGSORT_OFF(awkward_argsort_int32_uint32, int32_t, uint32_t)
ARGSORT_OFF(awkward_argsort_int32_uint64, int32_t, uint64_t)
ARGSORT_OFF(awkward_argsort_int32_int64, int32_t, int64_t)

ARGSORT_OFF(awkward_argsort_uint32_int32, uint32_t, int32_t)
ARGSORT_OFF(awkward_argsort_uint32_uint32, uint32_t, uint32_t)
ARGSORT_OFF(awkward_argsort_uint32_uint64, uint32_t, uint64_t)
ARGSORT_OFF(awkward_argsort_uint32_int64, uint32_t, int64_t)

ARGSORT_OFF(awkward_argsort_int64_int32, int64_t, int32_t)
ARGSORT_OFF(awkward_argsort_int64_uint32, int64_t, uint32_t)
ARGSORT_OFF(awkward_argsort_int64_uint64, int64_t, uint64_t)
ARGSORT_OFF(awkward_argsort_int64_int64, int64_t, int64_t)

ARGSORT_OFF(awkward_argsort_uint64_int32, uint64_t, int32_t)
ARGSORT_OFF(awkward_argsort_uint64_uint32, uint64_t, uint32_t)
ARGSORT_OFF(awkward_argsort_uint64_uint64, uint64_t, uint64_t)
ARGSORT_OFF(awkward_argsort_uint64_int64, uint64_t, int64_t)

ARGSORT_OFF(awkward_argsort_float32_int32, float, int32_t)
ARGSORT_OFF(awkward_argsort_float32_uint32, float, uint32_t)
ARGSORT_OFF(awkward_argsort_float32_uint64, float, uint64_t)
ARGSORT_OFF(awkward_argsort_float32_int64, float, int64_t)

ARGSORT_OFF(awkward_argsort_float64_int32, double, int32_t)
ARGSORT_OFF(awkward_argsort_float64_uint32, double, uint32_t)
ARGSORT_OFF(awkward_argsort_float64_uint64, double, uint64_t)
ARGSORT_OFF(awkward_argsort_float64_int64, double, int64_t)
