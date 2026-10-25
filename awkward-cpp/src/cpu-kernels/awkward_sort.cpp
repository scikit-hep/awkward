// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort.cpp", line)

#include "awkward/kernels.h"
#include "awkward/numeric-sort.h"

#include <algorithm>
#include <numeric>
#include <vector>

template <typename T>
ERROR awkward_sort(
  T* __restrict__ toptr,
  const T* __restrict__ fromptr,
  int64_t length,
  const int64_t* __restrict__ offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), int64_t{0});

  // Each segment of `index` is sorted independently.
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    // Clamp against `length` so malformed offsets cannot index past the end
    // of `index` (well-formed offsets satisfy offsets[last] <= length).
    int64_t start_off = static_cast<int64_t>(offsets[i]);
    int64_t stop_off = static_cast<int64_t>(offsets[i + 1]);
    if (start_off > length) start_off = length;
    if (stop_off > length) stop_off = length;
    if (start_off >= stop_off) continue;
    auto start = index.begin() + start_off;
    auto stop = index.begin() + stop_off;
    // Hoist the ascending/stable dispatch outside the per-element
    // comparisons (one branch per segment instead of O(n log n)).
    if (stable) {
      if (ascending) {
        std::stable_sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return awkward::kernel::NumericOrder<T, true>{}(fromptr[i1], fromptr[i2]);
        });
      } else {
        std::stable_sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return awkward::kernel::NumericOrder<T, false>{}(fromptr[i1], fromptr[i2]);
        });
      }
    } else {
      if (ascending) {
        std::sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return awkward::kernel::NumericOrder<T, true>{}(fromptr[i1], fromptr[i2]);
        });
      } else {
        std::sort(start, stop, [fromptr](int64_t i1, int64_t i2) {
          return awkward::kernel::NumericOrder<T, false>{}(fromptr[i1], fromptr[i2]);
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

#define SORT(FUNC, T)                                                      \
  ERROR FUNC(                                               \
    T* toptr, const T* fromptr, int64_t length,                            \
    const int64_t* offsets, int64_t offsetslength,                         \
    bool ascending, bool stable) {                                         \
    return awkward_sort<T>(                                                \
      toptr, fromptr, length, offsets, offsetslength,                      \
      ascending, stable);                                                  \
  }

SORT(awkward_sort_bool, bool)
SORT(awkward_sort_int8, int8_t)
SORT(awkward_sort_uint8, uint8_t)
SORT(awkward_sort_int16, int16_t)
SORT(awkward_sort_uint16, uint16_t)
SORT(awkward_sort_int32, int32_t)
SORT(awkward_sort_uint32, uint32_t)
SORT(awkward_sort_int64, int64_t)
SORT(awkward_sort_uint64, uint64_t)
SORT(awkward_sort_float32, float)
SORT(awkward_sort_float64, double)
