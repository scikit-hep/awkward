// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_nansum_count_complex.cpp", line)

#include "awkward/kernels.h"
#include <cmath>

template <typename T>
ERROR awkward_reduce_nansum_count_complex(
  T* __restrict__ toptr,
  int64_t* __restrict__ counts,
  const T* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    T real = static_cast<T>(0);
    T imag = static_cast<T>(0);
    int64_t count = 0;
    const int64_t stop = offsets[bin + 1];
    // Long segments check cache-local blocks, allowing the all-valid addition
    // loop to avoid a branch for every complex value. Short segments use
    // conditional additive identities instead of the extra block scan.
    // Accumulators are never reset between blocks: input addition order is
    // unchanged, including across NaN values and infinity cancellation.
    if (stop - offsets[bin] < 128) {
      for (int64_t i = offsets[bin]; i < stop; i++) {
        const T re = fromptr[2 * i];
        const T im = fromptr[2 * i + 1];
        const bool valid = !std::isnan(re) && !std::isnan(im);
        real += valid ? re : static_cast<T>(0);
        imag += valid ? im : static_cast<T>(0);
        count += valid;
      }
    } else {
      for (int64_t start = offsets[bin]; start < stop; start += 8) {
        const int64_t end = start + 8 < stop ? start + 8 : stop;
        bool valid = true;
        for (int64_t i = start; i < end; i++) {
          valid &= !std::isnan(fromptr[2 * i]) && !std::isnan(fromptr[2 * i + 1]);
        }
        if (valid) {
          for (int64_t i = start; i < end; i++) {
            real += fromptr[2 * i];
            imag += fromptr[2 * i + 1];
          }
          count += end - start;
        } else {
          for (int64_t i = start; i < end; i++) {
            const T re = fromptr[2 * i];
            const T im = fromptr[2 * i + 1];
            if (!std::isnan(re) && !std::isnan(im)) {
              real += re;
              imag += im;
              count++;
            }
          }
        }
      }
    }
    toptr[2 * bin] = real;
    toptr[2 * bin + 1] = imag;
    counts[bin] = count;
  }
  return success();
}

#define WRAPPER(FUNC, T) \
  ERROR FUNC(T* toptr, int64_t* counts, const T* fromptr, const int64_t* offsets, int64_t outlength) { \
    return awkward_reduce_nansum_count_complex<T>(toptr, counts, fromptr, offsets, outlength); \
  }

WRAPPER(awkward_reduce_nansum_count_complex64_complex64_64, float)
WRAPPER(awkward_reduce_nansum_count_complex128_complex128_64, double)
