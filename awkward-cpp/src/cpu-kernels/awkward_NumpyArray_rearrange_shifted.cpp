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
  int64_t offsetslength,
  const FROM* fromparents,
  const FROM* fromstarts) {
  int64_t k = 0;
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    for (int64_t j = 0; j < fromoffsets[i + 1] - fromoffsets[i]; j++) {
      toptr[k] = toptr[k] + fromoffsets[i];
      k++;
    }
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t parent = fromparents[i];
    int64_t start = fromstarts[parent];
    toptr[i] = toptr[i] + fromshifts[toptr[i]] - start;
  }

  return success();
}
ERROR
awkward_NumpyArray_rearrange_shifted_toint64_fromint64(
  int64_t* toptr,
  const int64_t* fromshifts,
  int64_t length,
  const int64_t* fromoffsets,
  int64_t offsetslength,
  const int64_t* fromparents,
  const int64_t* fromstarts) {
  return awkward_NumpyArray_rearrange_shifted<int64_t, int64_t>(
      toptr, fromshifts, length, fromoffsets, offsetslength, fromparents, fromstarts);
}
