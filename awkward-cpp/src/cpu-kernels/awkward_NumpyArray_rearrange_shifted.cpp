// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_rearrange_shifted.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_rearrange_shifted(
  TO* toptr,
  const FROM* fromshifts,
  int64_t length,
  const FROM* fromoffsets,
  int64_t outlength,
  const FROM* fromparents,
  const FROM* fromstarts) {
  // Phase 1: convert per-bin sorted positions into absolute positions by
  // adding fromoffsets[bin] to each element in bin's range. Walk bin-major.
  int64_t k = 0;
  for (int64_t bin = 0; bin < outlength; bin++) {
    FROM bin_offset = fromoffsets[bin];
    int64_t count = (int64_t)(fromoffsets[bin + 1] - bin_offset);
    for (int64_t j = 0; j < count; j++) {
      toptr[k] = (TO)(toptr[k] + bin_offset);
      k++;
    }
  }

  // Phase 2: apply per-shift correction. `length` is the number of shifts,
  // which is independent of the bin count, so we still need `fromparents`
  // to map each shift index to its outer bin.
  for (int64_t i = 0; i < length; i++) {
    FROM parent = fromparents[i];
    FROM start = fromstarts[parent];
    toptr[i] = (TO)(toptr[i] + fromshifts[toptr[i]] - start);
  }

  return success();
}
ERROR
awkward_NumpyArray_rearrange_shifted_toint64_fromint64(
  int64_t* toptr,
  const int64_t* fromshifts,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t outlength,
  const int64_t* fromparents,
  const int64_t* fromstarts) {
  return awkward_NumpyArray_rearrange_shifted<int64_t, int64_t>(
      toptr, fromshifts, length, fromoffsets, outlength, fromparents, fromstarts);
}
