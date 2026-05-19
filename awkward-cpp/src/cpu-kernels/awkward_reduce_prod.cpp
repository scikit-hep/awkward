// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_prod.cpp", line)

#include "awkward/kernels.h"

// Per-bin product. See `awkward_reduce_sum` for the rationale; the
// 4-accumulator unroll plus `__restrict__` qualifiers break the serial
// dependency on `acc *= ...` and let gcc/clang autovectorise the body.
// Multiplication is associative under integer wrap-around and floating-
// point reordering at the same precision class as numpy's own reduction
// (`ak.prod` has never guaranteed strictly-left-to-right order).
template <typename OUT, typename IN>
ERROR awkward_reduce_prod(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  #ifdef _OPENMP
  #pragma omp parallel for if(outlength > 1024) schedule(static)
  #endif
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    OUT a0 = static_cast<OUT>(1), a1 = static_cast<OUT>(1),
        a2 = static_cast<OUT>(1), a3 = static_cast<OUT>(1);
    int64_t i = start;
    for (; i + 4 <= stop; i += 4) {
      a0 *= static_cast<OUT>(fromptr[i + 0]);
      a1 *= static_cast<OUT>(fromptr[i + 1]);
      a2 *= static_cast<OUT>(fromptr[i + 2]);
      a3 *= static_cast<OUT>(fromptr[i + 3]);
    }
    OUT acc = (a0 * a1) * (a2 * a3);
    for (; i < stop; i++) {
      acc *= static_cast<OUT>(fromptr[i]);
    }
    toptr[bin] = acc;
  }
  return success();
}

#define REDUCE_PROD(OUT_T, OUT_N, IN_T, IN_N)                                         \
  ERROR awkward_reduce_prod_##OUT_N##_##IN_N##_64(                                    \
    OUT_T* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, int64_t outlength) {                                      \
    return awkward_reduce_prod<OUT_T, IN_T>(toptr, fromptr, offsets, outlength);      \
  }

REDUCE_PROD(int64_t,  int64,  int8_t,   int8)
REDUCE_PROD(uint64_t, uint64, uint8_t,  uint8)
REDUCE_PROD(int64_t,  int64,  int16_t,  int16)
REDUCE_PROD(uint64_t, uint64, uint16_t, uint16)
REDUCE_PROD(int64_t,  int64,  int32_t,  int32)
REDUCE_PROD(uint64_t, uint64, uint32_t, uint32)
REDUCE_PROD(int64_t,  int64,  int64_t,  int64)
REDUCE_PROD(uint64_t, uint64, uint64_t, uint64)
REDUCE_PROD(float,    float32, float,   float32)
REDUCE_PROD(double,   float64, double,  float64)
REDUCE_PROD(int32_t,  int32,  int8_t,   int8)
REDUCE_PROD(uint32_t, uint32, uint8_t,  uint8)
REDUCE_PROD(int32_t,  int32,  int16_t,  int16)
REDUCE_PROD(uint32_t, uint32, uint16_t, uint16)
REDUCE_PROD(int32_t,  int32,  int32_t,  int32)
REDUCE_PROD(uint32_t, uint32, uint32_t, uint32)
