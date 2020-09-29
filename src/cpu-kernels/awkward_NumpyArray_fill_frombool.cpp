// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill_frombool.cpp", line)

#include "awkward/kernels.h"

template <typename TO>
ERROR
awkward_NumpyArray_fill_frombool(TO* toptr,
                                 int64_t tooffset,
                                 const bool* fromptr,
                                 int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = (TO)(fromptr[i] ? 1 : 0);
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_tobool_frombool(bool* toptr,
                                        int64_t tooffset,
                                        const bool* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill_frombool<bool>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint8_frombool(int8_t* toptr,
                                        int64_t tooffset,
                                        const bool* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill_frombool<int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_frombool(int16_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_frombool(int32_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_frombool(int64_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_frombool(uint8_t* toptr,
                                         int64_t tooffset,
                                         const bool* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_frombool(uint16_t* toptr,
                                          int64_t tooffset,
                                          const bool* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_frombool(uint32_t* toptr,
                                          int64_t tooffset,
                                          const bool* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_frombool(uint64_t* toptr,
                                          int64_t tooffset,
                                          const bool* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_frombool<uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_frombool(float* toptr,
                                           int64_t tooffset,
                                           const bool* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_frombool<float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_frombool(double* toptr,
                                           int64_t tooffset,
                                           const bool* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_frombool<double>(
      toptr, tooffset, fromptr, length);
}
