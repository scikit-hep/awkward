// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmax.cpp", line)

#include "awkward/kernels.h"

// Per-bin argmax. See `awkward_reduce_argmin` for the rationale.
template <typename OUT, typename IN>
ERROR awkward_reduce_argmax(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  const int64_t* /* starts */,
  int64_t outlength) {
  #ifdef _OPENMP
  #pragma omp parallel for if(outlength > 1024) schedule(static)
  #endif
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    int64_t best = -1;
    if (start < stop) {
      best = start;
      IN best_val = fromptr[start];
      for (int64_t i = start + 1; i < stop; i++) {
        IN v = fromptr[i];
        if (v > best_val) {
          best_val = v;
          best = i;
        }
      }
    }
    toptr[bin] = static_cast<OUT>(best);
  }
  return success();
}

#define REDUCE_ARGMAX(IN_T, IN_N)                                                       \
  ERROR awkward_reduce_argmax_##IN_N##_64(                                              \
    int64_t* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, const int64_t* starts, int64_t outlength) {                 \
    return awkward_reduce_argmax<int64_t, IN_T>(                                        \
      toptr, fromptr, offsets, starts, outlength);                                      \
  }

REDUCE_ARGMAX(int8_t,   int8)
REDUCE_ARGMAX(uint8_t,  uint8)
REDUCE_ARGMAX(int16_t,  int16)
REDUCE_ARGMAX(uint16_t, uint16)
REDUCE_ARGMAX(int32_t,  int32)
REDUCE_ARGMAX(uint32_t, uint32)
REDUCE_ARGMAX(int64_t,  int64)
REDUCE_ARGMAX(uint64_t, uint64)
REDUCE_ARGMAX(float,    float32)
REDUCE_ARGMAX(double,   float64)
