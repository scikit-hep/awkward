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

#define REDUCE_PROD(FUNC, OUT_T, IN_T)                                         \
  ERROR FUNC(                                    \
    OUT_T* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, int64_t outlength) {                                      \
    return awkward_reduce_prod<OUT_T, IN_T>(toptr, fromptr, offsets, outlength);      \
  }

REDUCE_PROD(awkward_reduce_prod_int64_int8_64, int64_t, int8_t)
REDUCE_PROD(awkward_reduce_prod_uint64_uint8_64, uint64_t, uint8_t)
REDUCE_PROD(awkward_reduce_prod_int64_int16_64, int64_t, int16_t)
REDUCE_PROD(awkward_reduce_prod_uint64_uint16_64, uint64_t, uint16_t)
REDUCE_PROD(awkward_reduce_prod_int64_int32_64, int64_t, int32_t)
REDUCE_PROD(awkward_reduce_prod_uint64_uint32_64, uint64_t, uint32_t)
REDUCE_PROD(awkward_reduce_prod_int64_int64_64, int64_t, int64_t)
REDUCE_PROD(awkward_reduce_prod_uint64_uint64_64, uint64_t, uint64_t)
REDUCE_PROD(awkward_reduce_prod_float32_float32_64, float, float)
REDUCE_PROD(awkward_reduce_prod_float64_float64_64, double, double)
REDUCE_PROD(awkward_reduce_prod_int32_int8_64, int32_t, int8_t)
REDUCE_PROD(awkward_reduce_prod_uint32_uint8_64, uint32_t, uint8_t)
REDUCE_PROD(awkward_reduce_prod_int32_int16_64, int32_t, int16_t)
REDUCE_PROD(awkward_reduce_prod_uint32_uint16_64, uint32_t, uint16_t)
REDUCE_PROD(awkward_reduce_prod_int32_int32_64, int32_t, int32_t)
REDUCE_PROD(awkward_reduce_prod_uint32_uint32_64, uint32_t, uint32_t)
