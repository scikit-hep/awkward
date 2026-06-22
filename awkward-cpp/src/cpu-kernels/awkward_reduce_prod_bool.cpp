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

#define REDUCE_PROD_BOOL(IN_T, IN_N)                                                     \
  ERROR awkward_reduce_prod_bool_##IN_N##_64(                                            \
    bool* toptr, const IN_T* fromptr,                                                    \
    const int64_t* offsets, int64_t outlength) {                                         \
    return awkward_reduce_prod_bool<IN_T>(toptr, fromptr, offsets, outlength);           \
  }

REDUCE_PROD_BOOL(bool,     bool)
REDUCE_PROD_BOOL(int8_t,   int8)
REDUCE_PROD_BOOL(uint8_t,  uint8)
REDUCE_PROD_BOOL(int16_t,  int16)
REDUCE_PROD_BOOL(uint16_t, uint16)
REDUCE_PROD_BOOL(int32_t,  int32)
REDUCE_PROD_BOOL(uint32_t, uint32)
REDUCE_PROD_BOOL(int64_t,  int64)
REDUCE_PROD_BOOL(uint64_t, uint64)
REDUCE_PROD_BOOL(float,    float32)
REDUCE_PROD_BOOL(double,   float64)
