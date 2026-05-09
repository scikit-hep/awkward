// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_min.cpp", line)

#include "awkward/kernels.h"

// Per-bin minimum.
//
// Speed notes (calibrated by microbench):
//   * The 4-way accumulator unroll that helps `sum`/`prod` actively
//     hurts `min`/`max` on narrow bins because the per-bin combine
//     epilogue is paid once per bin, and on random data the original
//     `if (x < best)` already gets a well-predicted branch (branch
//     barely fires after `best` settles). Keep the implementation
//     simple — branchless `?:` plus `__restrict__` plus bound hoist —
//     which is Pareto-positive across the matrix we tested.
//   * `__restrict__` lets the compiler hoist `offsets[bin + 1]` out
//     of the inner loop instead of reloading on every iteration.
//   * The branchless `(v < best) ? v : best` form lowers to a SIMD
//     min when the autovectoriser fires (e.g. `vminps`/`vminpd` on
//     floats), and to a `cmov` otherwise.
//   * NaN semantics unchanged: floating-point `<` returns false for
//     NaN, so `best` stays — matching the original `if` form (and
//     `np.fmin`'s NaN-skipping behaviour).
template <typename OUT, typename IN>
ERROR awkward_reduce_min(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength,
  OUT identity) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    OUT best = identity;
    for (int64_t i = start; i < stop; i++) {
      OUT v = static_cast<OUT>(fromptr[i]);
      best = (v < best) ? v : best;
    }
    toptr[bin] = best;
  }
  return success();
}

#define REDUCE_MIN(T, NAME)                                                          \
  ERROR awkward_reduce_min_##NAME##_##NAME##_64(                                     \
    T* toptr, const T* fromptr,                                                      \
    const int64_t* offsets, int64_t outlength, T identity) {                         \
    return awkward_reduce_min<T, T>(toptr, fromptr, offsets, outlength, identity);   \
  }

REDUCE_MIN(int8_t,   int8)
REDUCE_MIN(uint8_t,  uint8)
REDUCE_MIN(int16_t,  int16)
REDUCE_MIN(uint16_t, uint16)
REDUCE_MIN(int32_t,  int32)
REDUCE_MIN(uint32_t, uint32)
REDUCE_MIN(int64_t,  int64)
REDUCE_MIN(uint64_t, uint64)
REDUCE_MIN(float,    float32)
REDUCE_MIN(double,   float64)
