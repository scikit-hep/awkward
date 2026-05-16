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
  #ifdef _OPENMP
  #pragma omp parallel for if(outlength > 1024) schedule(static)
  #endif
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

#define REDUCE_COUNTNONZERO(IN_T, IN_N)                                                  \
  ERROR awkward_reduce_countnonzero_##IN_N##_64(                                         \
    int64_t* toptr, const IN_T* fromptr,                                                 \
    const int64_t* offsets, int64_t outlength) {                                         \
    return awkward_reduce_countnonzero<IN_T>(toptr, fromptr, offsets, outlength);        \
  }

REDUCE_COUNTNONZERO(bool,     bool)
REDUCE_COUNTNONZERO(int8_t,   int8)
REDUCE_COUNTNONZERO(uint8_t,  uint8)
REDUCE_COUNTNONZERO(int16_t,  int16)
REDUCE_COUNTNONZERO(uint16_t, uint16)
REDUCE_COUNTNONZERO(int32_t,  int32)
REDUCE_COUNTNONZERO(uint32_t, uint32)
REDUCE_COUNTNONZERO(int64_t,  int64)
REDUCE_COUNTNONZERO(uint64_t, uint64)
REDUCE_COUNTNONZERO(float,    float32)
REDUCE_COUNTNONZERO(double,   float64)
