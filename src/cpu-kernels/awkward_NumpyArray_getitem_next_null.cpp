// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_getitem_next_null.cpp", line)

#include "awkward/kernels.h"

template <typename T>
ERROR awkward_NumpyArray_getitem_next_null(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len,
  int64_t stride,
  const T* pos) {
  uint8_t* end = toptr + len*stride;
  switch (stride) {
    case 1:
        while (toptr != end) *(toptr++) = fromptr[*(pos++)*stride];
        break;
    case 2:
        while (toptr != end) {
            std::memcpy(toptr, &fromptr[*(pos++)*stride], 2);
            toptr += 2;
        }
        break;
    case 4:
        while (toptr != end) {
            std::memcpy(toptr, &fromptr[*(pos++)*stride], 4);
            toptr += 4;
        }
        break;
    case 8:
        while (toptr != end) {
            std::memcpy(toptr, &fromptr[*(pos++)*stride], 8);
            toptr += 8;
        }
        break;
    default:
      while (toptr != end) {
          std::memcpy(toptr, &fromptr[*(pos++)*stride], (size_t)stride);
          toptr += stride;
      }
  }
  return success();
}
ERROR awkward_NumpyArray_getitem_next_null_64(
  uint8_t* toptr,
  const uint8_t* fromptr,
  int64_t len,
  int64_t stride,
  const int64_t* pos) {
  return awkward_NumpyArray_getitem_next_null(
    toptr,
    fromptr,
    len,
    stride,
    pos);
}
