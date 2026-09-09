// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sumofsquares.cpp", line)

#include "awkward/kernels.h"

// Per-bin sum of squares. The input element is widened to the (floating) output
// type *before* squaring, so the multiply happens in `OUT` and never overflows
// the input type -- and no intermediate `x*x` buffer is materialised. Matches
// the kernel-specification.yml reference exactly.
template <typename OUT, typename IN>
ERROR awkward_reduce_sumofsquares(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT acc = OUT{};
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      OUT v = static_cast<OUT>(fromptr[i]);
      acc += v * v;
    }
    toptr[bin] = acc;
  }
  return success();
}

#define REDUCE_SUMSQ(FUNC, OUT_T, IN_T)                                       \
  ERROR FUNC(                                                                 \
    OUT_T* toptr, const IN_T* fromptr,                                        \
    const int64_t* offsets, int64_t outlength) {                             \
    return awkward_reduce_sumofsquares<OUT_T, IN_T>(                          \
      toptr, fromptr, offsets, outlength);                                    \
  }

REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_int8_64, double, int8_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_uint8_64, double, uint8_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_int16_64, double, int16_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_uint16_64, double, uint16_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_int32_64, double, int32_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_uint32_64, double, uint32_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_int64_64, double, int64_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_uint64_64, double, uint64_t)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_bool_64, double, bool)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_float32_64, double, float)
REDUCE_SUMSQ(awkward_reduce_sumofsquares_float64_float64_64, double, double)
