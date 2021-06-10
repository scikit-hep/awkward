// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill_scaled.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill_scaled(TO* toptr,
                               int64_t tooffset,
                               const FROM* fromptr,
                               int64_t length,
                               double scale) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = (TO)(fromptr[i] * scale);
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_scaled_toint64_fromint64(int64_t* toptr,
                                                 int64_t tooffset,
                                                 const int64_t* fromptr,
                                                 int64_t length,
                                                 double scale) {
  return awkward_NumpyArray_fill_scaled<int64_t, int64_t>(
      toptr, tooffset, fromptr, length, scale);
}
