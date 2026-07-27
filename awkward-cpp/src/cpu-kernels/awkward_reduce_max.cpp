// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_max.cpp", line)

#include "awkward/kernels.h"

// Per-bin maximum. See `awkward_reduce_min` for the rationale.
template <typename OUT, typename IN>
ERROR awkward_reduce_max(
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
      best = (v > best) ? v : best;
    }
    toptr[bin] = best;
  }
  return success();
}

#define REDUCE_MAX(FUNC, T)                                                          \
  ERROR FUNC(                                     \
    T* toptr, const T* fromptr,                                                      \
    const int64_t* offsets, int64_t outlength, T identity) {                         \
    return awkward_reduce_max<T, T>(toptr, fromptr, offsets, outlength, identity);   \
  }

REDUCE_MAX(awkward_reduce_max_int8_int8_64, int8_t)
REDUCE_MAX(awkward_reduce_max_uint8_uint8_64, uint8_t)
REDUCE_MAX(awkward_reduce_max_int16_int16_64, int16_t)
REDUCE_MAX(awkward_reduce_max_uint16_uint16_64, uint16_t)
REDUCE_MAX(awkward_reduce_max_int32_int32_64, int32_t)
REDUCE_MAX(awkward_reduce_max_uint32_uint32_64, uint32_t)
REDUCE_MAX(awkward_reduce_max_int64_int64_64, int64_t)
REDUCE_MAX(awkward_reduce_max_uint64_uint64_64, uint64_t)
REDUCE_MAX(awkward_reduce_max_float32_float32_64, float)
REDUCE_MAX(awkward_reduce_max_float64_float64_64, double)
