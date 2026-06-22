// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_argmin.cpp", line)

#include "awkward/kernels.h"

// Per-bin argmin.
//
// Speed notes (calibrated against the family microbench):
//   * The inner comparison is data-dependent — `best` chains through
//     every iteration, and the original `fromptr[i] < fromptr[best]`
//     load reaches into a moving target. Vectorising it isn't viable
//     so we keep the loop simple, but we do two things to help the
//     plain serial version:
//       (a) hoist `fromptr[best]` into a register `best_val` so each
//           iteration is one load + one compare + one mispredictable
//           branch, instead of two loads + ALU on `best`.
//       (b) peel the first iteration out of the inner loop so the
//           hot path no longer pays the `best == -1` short-circuit
//           every step.
//   * `__restrict__` lets the compiler hoist `offsets[bin + 1]`.
//   * NaN semantics unchanged: `<` returns false for NaN, so `best`
//     stays put when either side is NaN — matches the original `if`.
template <typename OUT, typename IN>
ERROR awkward_reduce_argmin(
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
        if (v < best_val) {
          best_val = v;
          best = i;
        }
      }
    }
    toptr[bin] = static_cast<OUT>(best);
  }
  return success();
}

#define REDUCE_ARGMIN(IN_T, IN_N)                                                       \
  ERROR awkward_reduce_argmin_##IN_N##_64(                                              \
    int64_t* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, const int64_t* starts, int64_t outlength) {                 \
    return awkward_reduce_argmin<int64_t, IN_T>(                                        \
      toptr, fromptr, offsets, starts, outlength);                                      \
  }

REDUCE_ARGMIN(int8_t,   int8)
REDUCE_ARGMIN(uint8_t,  uint8)
REDUCE_ARGMIN(int16_t,  int16)
REDUCE_ARGMIN(uint16_t, uint16)
REDUCE_ARGMIN(int32_t,  int32)
REDUCE_ARGMIN(uint32_t, uint32)
REDUCE_ARGMIN(int64_t,  int64)
REDUCE_ARGMIN(uint64_t, uint64)
REDUCE_ARGMIN(float,    float32)
REDUCE_ARGMIN(double,   float64)
