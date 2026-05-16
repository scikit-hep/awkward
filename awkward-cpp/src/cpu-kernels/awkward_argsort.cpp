// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_argsort.cpp", line)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

#include "awkward/kernels.h"

// Per-segment argsort. NaNs are pushed to the high end (matching NumPy /
// the older hand-rolled comparator). The ascending/descending choice and
// the floating-point NaN handling are folded into a single inline lambda;
// `if constexpr` lets the bool / integer specialisations compile without
// an `std::isnan(bool)` instantiation.
template <typename T, typename U>
ERROR awkward_argsort(
  int64_t* toptr,
  const T* fromptr,
  int64_t length,
  const U* offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::iota(toptr, toptr + length, int64_t{0});

  auto less = [&](int64_t a, int64_t b) -> bool {
    T l = fromptr[a];
    T r = fromptr[b];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(r)) return false;
      if (std::isnan(l)) return true;
    }
    return ascending ? (l < r) : (l > r);
  };

  // Bin loop is embarrassingly parallel: each segment of `toptr` is sorted
  // independently. The `if(...)` clause keeps the parallel region inert for
  // tiny outputs where thread-startup would dominate.
  #ifdef _OPENMP
  #pragma omp parallel for if(offsetslength > 1024) schedule(dynamic, 64)
  #endif
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    int64_t lo = static_cast<int64_t>(offsets[i]);
    int64_t hi = static_cast<int64_t>(offsets[i + 1]);
    int64_t* first = toptr + lo;
    int64_t* last  = toptr + hi;

    if (stable) std::stable_sort(first, last, less);
    else        std::sort(first, last, less);

    std::transform(first, last, first,
                   [lo](int64_t j) { return j - lo; });
  }
  return success();
}

#define ARGSORT_OFF(T, NAME, U, ONAME)                                      \
  ERROR awkward_argsort_##NAME##_##ONAME(                                   \
    int64_t* toptr, const T* fromptr, int64_t length,                       \
    const U* offsets, int64_t offsetslength,                                \
    bool ascending, bool stable) {                                          \
    return awkward_argsort<T, U>(                                           \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);   \
  }

#define ARGSORT(T, NAME)                  \
  ARGSORT_OFF(T, NAME, int32_t,  int32)   \
  ARGSORT_OFF(T, NAME, uint32_t, uint32)  \
  ARGSORT_OFF(T, NAME, uint64_t, uint64)  \
  ARGSORT_OFF(T, NAME, int64_t,  int64)

ARGSORT(bool,     bool)
ARGSORT(int8_t,   int8)
ARGSORT(uint8_t,  uint8)
ARGSORT(int16_t,  int16)
ARGSORT(uint16_t, uint16)
ARGSORT(int32_t,  int32)
ARGSORT(uint32_t, uint32)
ARGSORT(int64_t,  int64)
ARGSORT(uint64_t, uint64)
ARGSORT(float,    float32)
ARGSORT(double,   float64)
