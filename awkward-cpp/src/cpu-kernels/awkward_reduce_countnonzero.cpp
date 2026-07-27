// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_countnonzero.cpp", line)

#include "awkward/kernels.h"

// Per-bin count of non-zero elements. See `awkward_reduce_sum` for the
// rationale; four independent counters break the dependency chain on
// `c++` and let gcc/clang autovectorise the body into masked SIMD adds
// (`vpcmpneqd` + `vpaddd`-style sequences). The conditional increment is
// rewritten branchlessly as `c += (x != 0)` so the vectoriser doesn't
// have to prove it can hoist the predicate.
template <typename IN>
ERROR awkward_reduce_countnonzero(
  int64_t* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    int64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int64_t i = start;
    for (; i + 4 <= stop; i += 4) {
      c0 += (fromptr[i + 0] != 0);
      c1 += (fromptr[i + 1] != 0);
      c2 += (fromptr[i + 2] != 0);
      c3 += (fromptr[i + 3] != 0);
    }
    int64_t c = (c0 + c1) + (c2 + c3);
    for (; i < stop; i++) {
      c += (fromptr[i] != 0);
    }
    toptr[bin] = c;
  }
  return success();
}

#define REDUCE_COUNTNONZERO(FUNC, IN_T)                                                  \
  ERROR FUNC(                                         \
    int64_t* toptr, const IN_T* fromptr,                                                 \
    const int64_t* offsets, int64_t outlength) {                                         \
    return awkward_reduce_countnonzero<IN_T>(toptr, fromptr, offsets, outlength);        \
  }

REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_bool_64, bool)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_int8_64, int8_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_uint8_64, uint8_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_int16_64, int16_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_uint16_64, uint16_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_int32_64, int32_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_uint32_64, uint32_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_int64_64, int64_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_uint64_64, uint64_t)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_float32_64, float)
REDUCE_COUNTNONZERO(awkward_reduce_countnonzero_float64_64, double)
