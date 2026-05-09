// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum_bool.cpp", line)

#include "awkward/kernels.h"

// Per-bin "any nonzero" reduction.
//
// We deliberately keep the early-exit: realistic data has nonzeros
// scattered throughout, so on average we exit far before the end of
// the bin; turning this into a full scan to enable vectorisation
// would lose more than it gains for the common case.
//
// `__restrict__` lets the compiler hoist `offsets[bin + 1]` out of the
// inner loop bound check (without it the standard pessimistically
// reloads it every iteration in case `toptr` writes alias `offsets`).
template <typename IN>
ERROR awkward_reduce_sum_bool(
  bool* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    bool found = false;
    for (int64_t i = start; i < stop; i++) {
      if (fromptr[i] != 0) { found = true; break; }
    }
    toptr[bin] = found;
  }
  return success();
}

#define REDUCE_SUM_BOOL(IN_T, IN_N)                                                      \
  ERROR awkward_reduce_sum_bool_##IN_N##_64(                                             \
    bool* toptr, const IN_T* fromptr,                                                    \
    const int64_t* offsets, int64_t outlength) {                                         \
    return awkward_reduce_sum_bool<IN_T>(toptr, fromptr, offsets, outlength);            \
  }

REDUCE_SUM_BOOL(bool,     bool)
REDUCE_SUM_BOOL(int8_t,   int8)
REDUCE_SUM_BOOL(uint8_t,  uint8)
REDUCE_SUM_BOOL(int16_t,  int16)
REDUCE_SUM_BOOL(uint16_t, uint16)
REDUCE_SUM_BOOL(int32_t,  int32)
REDUCE_SUM_BOOL(uint32_t, uint32)
REDUCE_SUM_BOOL(int64_t,  int64)
REDUCE_SUM_BOOL(uint64_t, uint64)
REDUCE_SUM_BOOL(float,    float32)
REDUCE_SUM_BOOL(double,   float64)
