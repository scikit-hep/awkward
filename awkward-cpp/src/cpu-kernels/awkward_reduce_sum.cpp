// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_sum.cpp", line)

#include "awkward/kernels.h"

// Per-bin sum: serial left-to-right accumulation, matching the
// kernel-specification.yml reference exactly. `__restrict__` lets the
// compiler hoist `offsets[bin + 1]` and autovectorise the (associative)
// integer cases on its own.
template <typename OUT, typename IN>
ERROR awkward_reduce_sum(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    OUT acc = OUT{};
    for (int64_t i = offsets[bin]; i < offsets[bin + 1]; i++) {
      acc += static_cast<OUT>(fromptr[i]);
    }
    toptr[bin] = acc;
  }
  return success();
}

#define REDUCE_SUM(FUNC, OUT_T, IN_T)                                          \
  ERROR FUNC(                                     \
    OUT_T* toptr, const IN_T* fromptr,                                                \
    const int64_t* offsets, int64_t outlength) {                                      \
    return awkward_reduce_sum<OUT_T, IN_T>(toptr, fromptr, offsets, outlength);       \
  }

REDUCE_SUM(awkward_reduce_sum_int64_int8_64, int64_t, int8_t)
REDUCE_SUM(awkward_reduce_sum_uint64_uint8_64, uint64_t, uint8_t)
REDUCE_SUM(awkward_reduce_sum_int64_int16_64, int64_t, int16_t)
REDUCE_SUM(awkward_reduce_sum_uint64_uint16_64, uint64_t, uint16_t)
REDUCE_SUM(awkward_reduce_sum_int64_int32_64, int64_t, int32_t)
REDUCE_SUM(awkward_reduce_sum_uint64_uint32_64, uint64_t, uint32_t)
REDUCE_SUM(awkward_reduce_sum_int64_int64_64, int64_t, int64_t)
REDUCE_SUM(awkward_reduce_sum_uint64_uint64_64, uint64_t, uint64_t)
REDUCE_SUM(awkward_reduce_sum_float32_float32_64, float, float)
REDUCE_SUM(awkward_reduce_sum_float64_float64_64, double, double)
REDUCE_SUM(awkward_reduce_sum_int32_int8_64, int32_t, int8_t)
REDUCE_SUM(awkward_reduce_sum_uint32_uint8_64, uint32_t, uint8_t)
REDUCE_SUM(awkward_reduce_sum_int32_int16_64, int32_t, int16_t)
REDUCE_SUM(awkward_reduce_sum_uint32_uint16_64, uint32_t, uint16_t)
REDUCE_SUM(awkward_reduce_sum_int32_int32_64, int32_t, int32_t)
REDUCE_SUM(awkward_reduce_sum_uint32_uint32_64, uint32_t, uint32_t)
