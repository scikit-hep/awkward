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

#define REDUCE_ARGMAX(FUNC, IN_T)                                                       \
  ERROR FUNC(                                              \
    int64_t* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, const int64_t* starts, int64_t outlength) {                 \
    return awkward_reduce_argmax<int64_t, IN_T>(                                        \
      toptr, fromptr, offsets, starts, outlength);                                      \
  }

REDUCE_ARGMAX(awkward_reduce_argmax_int8_64, int8_t)
REDUCE_ARGMAX(awkward_reduce_argmax_uint8_64, uint8_t)
REDUCE_ARGMAX(awkward_reduce_argmax_int16_64, int16_t)
REDUCE_ARGMAX(awkward_reduce_argmax_uint16_64, uint16_t)
REDUCE_ARGMAX(awkward_reduce_argmax_int32_64, int32_t)
REDUCE_ARGMAX(awkward_reduce_argmax_uint32_64, uint32_t)
REDUCE_ARGMAX(awkward_reduce_argmax_int64_64, int64_t)
REDUCE_ARGMAX(awkward_reduce_argmax_uint64_64, uint64_t)
REDUCE_ARGMAX(awkward_reduce_argmax_float32_64, float)
REDUCE_ARGMAX(awkward_reduce_argmax_float64_64, double)
