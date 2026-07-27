// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sumofpowers.cpp", line)

#include <cmath>

#include "awkward/kernels.h"

// x**n by exponentiation-by-squaring for a non-negative integer n. This is what
// ak.moment passes (n is a small whole number), and it is far cheaper than the
// generic std::pow(x, (double)n) -- a couple of multiplies instead of a
// transcendental call per element. Falls back to std::pow only for the unusual
// negative n. Computed in the (double) output type: no overflow/precision loss.
template <typename OUT>
inline OUT ipow(OUT base, int64_t n) {
  if (n < 0) {
    return std::pow(base, static_cast<OUT>(n));
  }
  OUT result = OUT{1};
  OUT b = base;
  int64_t e = n;
  while (e > 0) {
    if (e & 1) {
      result *= b;
    }
    e >>= 1;
    if (e > 0) {
      b *= b;
    }
  }
  return result;
}

// Per-bin sum of n-th powers. Each element is widened to the (floating) output
// type before raising to the power, so integer/float32 powers accumulate in
// double precision -- no overflow/precision loss and no intermediate x**n
// buffer. `n` is a runtime argument, so one kernel covers every power.
template <typename OUT, typename IN>
ERROR awkward_reduce_sumofpowers(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength,
  int64_t n) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT acc = OUT{};
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      acc += ipow(static_cast<OUT>(fromptr[i]), n);
    }
    toptr[bin] = acc;
  }
  return success();
}

#define REDUCE_SUMPOW(FUNC, OUT_T, IN_T)                                       \
  ERROR FUNC(                                                                  \
    OUT_T* toptr, const IN_T* fromptr,                                         \
    const int64_t* offsets, int64_t outlength, int64_t n) {                   \
    return awkward_reduce_sumofpowers<OUT_T, IN_T>(                           \
      toptr, fromptr, offsets, outlength, n);                                  \
  }

REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_int8_64, double, int8_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_uint8_64, double, uint8_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_int16_64, double, int16_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_uint16_64, double, uint16_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_int32_64, double, int32_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_uint32_64, double, uint32_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_int64_64, double, int64_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_uint64_64, double, uint64_t)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_bool_64, double, bool)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_float32_64, double, float)
REDUCE_SUMPOW(awkward_reduce_sumofpowers_float64_float64_64, double, double)
