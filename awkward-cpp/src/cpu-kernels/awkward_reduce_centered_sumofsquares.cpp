// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_centered_sumofsquares.cpp", line)

#include "awkward/kernels.h"

// Per-bin sum of squared deviations: Sigma (x - mean)**2, the numerator of the
// two-pass variance. `means` holds one (float64) mean per output bin, aligned to
// `offsets`/`outlength` (produced by the same reduce descent as the sum, so bin
// order matches). Each element is widened to the (double) output type before
// centring, so integer/float32 inputs neither overflow nor lose precision, and
// no (x - mean) deviation buffer is materialised. For an empty bin means[bin] is
// still loaded (it is in bounds), but unused, since the inner loop runs zero times.
template <typename OUT, typename IN>
ERROR awkward_reduce_centered_sumofsquares(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength,
  const OUT* __restrict__ means) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT acc = OUT{};
    OUT m = means[bin];
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      OUT d = static_cast<OUT>(fromptr[i]) - m;
      acc += d * d;
    }
    toptr[bin] = acc;
  }
  return success();
}

#define REDUCE_CENTERED_SUMSQ(FUNC, OUT_T, IN_T)                               \
  ERROR FUNC(                                                                  \
    OUT_T* toptr, const IN_T* fromptr,                                         \
    const int64_t* offsets, int64_t outlength, const OUT_T* means) {           \
    return awkward_reduce_centered_sumofsquares<OUT_T, IN_T>(                  \
      toptr, fromptr, offsets, outlength, means);                             \
  }

REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_int8_64, double, int8_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_uint8_64, double, uint8_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_int16_64, double, int16_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_uint16_64, double, uint16_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_int32_64, double, int32_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_uint32_64, double, uint32_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_int64_64, double, int64_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_uint64_64, double, uint64_t)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_bool_64, double, bool)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_float32_64, double, float)
REDUCE_CENTERED_SUMSQ(awkward_reduce_centered_sumofsquares_float64_float64_64, double, double)
