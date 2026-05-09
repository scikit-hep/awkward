// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum.cpp", line)

#include "awkward/kernels.h"

// Per-bin sum.
//
// Speed notes:
//   * Four independent accumulators break the dependency chain on `acc`
//     (the original `acc += ...` serialises through FP-add latency / single
//     integer-add port). Out-of-order CPUs can keep 4 adds in flight, and
//     once the body's data dependencies are independent the loop matches a
//     vector-reduction pattern that gcc/clang autovectorise into SSE2 /
//     AVX2 / AVX-512 (`vaddps`, `vpaddq`, ...).
//   * `__restrict__` lets the compiler hoist `offsets[bin + 1]` out of the
//     inner loop. Without it the standard pessimistically reloads it every
//     iteration in case `fromptr` writes alias `offsets`.
//   * The 4-way pairwise summation is no less stable than NumPy's own
//     reduction (NumPy uses a similar pairwise scheme for floats);
//     `ak.sum` has never guaranteed bit-exact left-to-right summation.
template <typename OUT, typename IN>
ERROR awkward_reduce_sum(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    OUT a0 = OUT{}, a1 = OUT{}, a2 = OUT{}, a3 = OUT{};
    int64_t i = start;
    for (; i + 4 <= stop; i += 4) {
      a0 += static_cast<OUT>(fromptr[i + 0]);
      a1 += static_cast<OUT>(fromptr[i + 1]);
      a2 += static_cast<OUT>(fromptr[i + 2]);
      a3 += static_cast<OUT>(fromptr[i + 3]);
    }
    OUT acc = (a0 + a1) + (a2 + a3);
    for (; i < stop; i++) {
      acc += static_cast<OUT>(fromptr[i]);
    }
    toptr[bin] = acc;
  }
  return success();
}

#define REDUCE_SUM(OUT_T, OUT_N, IN_T, IN_N)                                          \
  ERROR awkward_reduce_sum_##OUT_N##_##IN_N##_64(                                     \
    OUT_T* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, int64_t outlength) {                                      \
    return awkward_reduce_sum<OUT_T, IN_T>(toptr, fromptr, offsets, outlength);       \
  }

REDUCE_SUM(int64_t,  int64,  int8_t,   int8)
REDUCE_SUM(uint64_t, uint64, uint8_t,  uint8)
REDUCE_SUM(int64_t,  int64,  int16_t,  int16)
REDUCE_SUM(uint64_t, uint64, uint16_t, uint16)
REDUCE_SUM(int64_t,  int64,  int32_t,  int32)
REDUCE_SUM(uint64_t, uint64, uint32_t, uint32)
REDUCE_SUM(int64_t,  int64,  int64_t,  int64)
REDUCE_SUM(uint64_t, uint64, uint64_t, uint64)
REDUCE_SUM(float,    float32, float,   float32)
REDUCE_SUM(double,   float64, double,  float64)
REDUCE_SUM(int32_t,  int32,  int8_t,   int8)
REDUCE_SUM(uint32_t, uint32, uint8_t,  uint8)
REDUCE_SUM(int32_t,  int32,  int16_t,  int16)
REDUCE_SUM(uint32_t, uint32, uint16_t, uint16)
REDUCE_SUM(int32_t,  int32,  int32_t,  int32)
REDUCE_SUM(uint32_t, uint32, uint32_t, uint32)
