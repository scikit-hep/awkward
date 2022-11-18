// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_rearrange_shifted.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_rearrange_shifted(TO* toptr,
                                     const FROM* shifts,
                                     int64_t length,
                                     const FROM* offsets,
                                     int64_t offsetslength,
                                     const FROM* parents,
                                     int64_t /* parentslength */,  // FIXME: these arguments are not needed
                                     const FROM* starts,
                                     int64_t /* startslength */) {
  int64_t k = 0;
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    for (int64_t j = 0; j < offsets[i + 1] - offsets[i]; j++) {
      toptr[k] = toptr[k] + offsets[i];
      k++;
    }
  }
  for (int64_t i = 0;  i < length;  i++) {
    int64_t parent = parents[i];
    int64_t start = starts[parent];
    toptr[i] = toptr[i] + shifts[toptr[i]] - start;
  }

  return success();
}
ERROR
awkward_NumpyArray_rearrange_shifted_toint64_fromint64(int64_t* toptr,
                                                       const int64_t* fromshifts,
                                                       int64_t length,
                                                       const int64_t* fromoffsets,
                                                       int64_t offsetslength,
                                                       const int64_t* fromparents,
                                                       int64_t parentslength,
                                                       const int64_t* fromstarts,
                                                       int64_t startslength) {
  return awkward_NumpyArray_rearrange_shifted<int64_t, int64_t>(
      toptr, fromshifts, length, fromoffsets, offsetslength, fromparents, parentslength, fromstarts, startslength);
}
