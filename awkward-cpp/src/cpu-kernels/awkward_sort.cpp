// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort.cpp", line)

#include "awkward/kernels.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

// Per-segment sort by value. NaNs are pushed to the high end (matching
// NumPy / the older hand-rolled comparator). Direction and NaN handling
// fold into a single inline lambda; `if constexpr` lets the bool/integer
// specialisations skip the floating-point NaN branch entirely (so we
// never instantiate `std::isnan(bool)` and the explicit bool override
// helpers are no longer needed).
template <typename T>
ERROR awkward_sort(
  T* toptr,
  const T* fromptr,
  int64_t length,
  const int64_t* offsets,
  int64_t offsetslength,
  int64_t /* parentslength */,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), int64_t{0});

  auto less = [&](int64_t a, int64_t b) -> bool {
    T l = fromptr[a];
    T r = fromptr[b];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(r)) return false;
      if (std::isnan(l)) return true;
    }
    return ascending ? (l < r) : (l > r);
  };

  // Bin loop is embarrassingly parallel: each segment of `index` is sorted
  // independently. `dynamic` schedule helps when bin sizes are uneven.
  #ifdef _OPENMP
  #pragma omp parallel for if(offsetslength > 1024) schedule(dynamic, 64)
  #endif
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    auto first = index.begin() + offsets[i];
    auto last  = index.begin() + offsets[i + 1];
    if (stable) std::stable_sort(first, last, less);
    else        std::sort(first, last, less);
  }

  int64_t parentslength_eff = offsets[offsetslength - 1];
  int64_t copy_length = (parentslength_eff < length) ? parentslength_eff : length;
  for (int64_t i = 0; i < copy_length; i++) {
    toptr[i] = fromptr[index[i]];
  }
  return success();
}

#define SORT(T, NAME)                                                      \
  ERROR awkward_sort_##NAME(                                               \
    T* toptr, const T* fromptr, int64_t length,                            \
    const int64_t* offsets, int64_t offsetslength, int64_t parentslength,  \
    bool ascending, bool stable) {                                         \
    return awkward_sort<T>(                                                \
      toptr, fromptr, length, offsets, offsetslength,                      \
      parentslength, ascending, stable);                                   \
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
