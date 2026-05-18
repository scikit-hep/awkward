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
  #ifdef _OPENMP
  #pragma omp parallel for if(outlength > 1024) schedule(static)
  #endif
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

#define REDUCE_MAX(T, NAME)                                                          \
  ERROR awkward_reduce_max_##NAME##_##NAME##_64(                                     \
    T* toptr, const T* fromptr,                                                      \
    const int64_t* offsets, int64_t outlength, T identity) {                         \
    return awkward_reduce_max<T, T>(toptr, fromptr, offsets, outlength, identity);   \
  }

REDUCE_MAX(int8_t,   int8)
REDUCE_MAX(uint8_t,  uint8)
REDUCE_MAX(int16_t,  int16)
REDUCE_MAX(uint16_t, uint16)
REDUCE_MAX(int32_t,  int32)
REDUCE_MAX(uint32_t, uint32)
REDUCE_MAX(int64_t,  int64)
REDUCE_MAX(uint64_t, uint64)
REDUCE_MAX(float,    float32)
REDUCE_MAX(double,   float64)
