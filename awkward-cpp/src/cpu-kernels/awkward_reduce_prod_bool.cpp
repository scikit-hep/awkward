// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod_bool.cpp", line)

#include "awkward/kernels.h"

// Per-bin "all nonzero" reduction. See `awkward_reduce_sum_bool` for
// the rationale behind keeping the early-exit and adding `__restrict__`.
template <typename IN>
ERROR awkward_reduce_prod_bool(
  bool* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    bool all_nonzero = true;
    for (int64_t i = start; i < stop; i++) {
      if (fromptr[i] == 0) { all_nonzero = false; break; }
    }
    toptr[bin] = all_nonzero;
  }
  return success();
}

#define REDUCE_PROD_BOOL(FUNC, IN_T)                                                     \
  ERROR FUNC(                                            \
    bool* toptr, const IN_T* fromptr,                                                    \
    const int64_t* offsets, int64_t outlength) {                                         \
    return awkward_reduce_prod_bool<IN_T>(toptr, fromptr, offsets, outlength);           \
  }

REDUCE_PROD_BOOL(awkward_reduce_prod_bool_bool_64, bool)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_int8_64, int8_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_uint8_64, uint8_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_int16_64, int16_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_uint16_64, uint16_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_int32_64, int32_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_uint32_64, uint32_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_int64_64, int64_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_uint64_64, uint64_t)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_float32_64, float)
REDUCE_PROD_BOOL(awkward_reduce_prod_bool_float64_64, double)
