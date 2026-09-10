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

#define REDUCE_SUM_BOOL(FUNC, IN_T)                                                      \
  ERROR FUNC(                                             \
    bool* toptr, const IN_T* fromptr,                                                    \
    const int64_t* offsets, int64_t outlength) {                                         \
    return awkward_reduce_sum_bool<IN_T>(toptr, fromptr, offsets, outlength);            \
  }

REDUCE_SUM_BOOL(awkward_reduce_sum_bool_bool_64, bool)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_int8_64, int8_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_uint8_64, uint8_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_int16_64, int16_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_uint16_64, uint16_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_int32_64, int32_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_uint32_64, uint32_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_int64_64, int64_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_uint64_64, uint64_t)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_float32_64, float)
REDUCE_SUM_BOOL(awkward_reduce_sum_bool_float64_64, double)
