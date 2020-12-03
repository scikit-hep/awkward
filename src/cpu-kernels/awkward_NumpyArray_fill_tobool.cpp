// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill_tobool.cpp", line)

#include "awkward/kernels.h"

template <typename FROM>
ERROR
awkward_NumpyArray_fill_tobool(bool* toptr,
                               int64_t tooffset,
                               const FROM* fromptr,
                               int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = fromptr[i] > 0 ? true : false;
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_tobool_fromint8(bool* toptr,
                                        int64_t tooffset,
                                        const int8_t* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill_tobool<int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromint16(bool* toptr,
                                         int64_t tooffset,
                                         const int16_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromint32(bool* toptr,
                                         int64_t tooffset,
                                         const int32_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromint64(bool* toptr,
                                         int64_t tooffset,
                                         const int64_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint8(bool* toptr,
                                         int64_t tooffset,
                                         const uint8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint16(bool* toptr,
                                          int64_t tooffset,
                                          const uint16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint32(bool* toptr,
                                          int64_t tooffset,
                                          const uint32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromuint64(bool* toptr,
                                          int64_t tooffset,
                                          const uint64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill_tobool<uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromfloat32(bool* toptr,
                                           int64_t tooffset,
                                           const float* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_tobool<float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromfloat64(bool* toptr,
                                           int64_t tooffset,
                                           const double* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill_tobool<double>(
      toptr, tooffset, fromptr, length);
}
