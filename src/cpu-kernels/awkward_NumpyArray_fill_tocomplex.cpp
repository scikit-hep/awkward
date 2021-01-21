// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill_tocomplex.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill_tocomplex(TO* toptr,
                                  int64_t tooffset,
                                  const FROM* fromptr,
                                  int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + 2 * i] = (TO)fromptr[i];
    toptr[tooffset + 2 * i + 1] = (TO)0;
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_tocomplex64_frombool(float* toptr,
                                             int64_t tooffset,
                                             const bool* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<bool, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromint8(float* toptr,
                                             int64_t tooffset,
                                             const int8_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int8_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromint16(float* toptr,
                                              int64_t tooffset,
                                              const int16_t* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int16_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromint32(float* toptr,
                                              int64_t tooffset,
                                              const int32_t* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int32_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromint64(float* toptr,
                                              int64_t tooffset,
                                              const int64_t* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int64_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromuint8(float* toptr,
                                              int64_t tooffset,
                                              const uint8_t* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint8_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromuint16(float* toptr,
                                               int64_t tooffset,
                                               const uint16_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint16_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromuint32(float* toptr,
                                               int64_t tooffset,
                                               const uint32_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint32_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromuint64(float* toptr,
                                               int64_t tooffset,
                                               const uint64_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint64_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromfloat32(float* toptr,
                                                int64_t tooffset,
                                                const float* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<float, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex64_fromfloat64(float* toptr,
                                                int64_t tooffset,
                                                const double* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<double, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_frombool(double* toptr,
                                              int64_t tooffset,
                                              const bool* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<bool, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromint8(double* toptr,
                                              int64_t tooffset,
                                              const int8_t* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int8_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromint16(double* toptr,
                                               int64_t tooffset,
                                               const int16_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int16_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromint32(double* toptr,
                                               int64_t tooffset,
                                               const int32_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int32_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromint64(double* toptr,
                                               int64_t tooffset,
                                               const int64_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<int64_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromuint8(double* toptr,
                                               int64_t tooffset,
                                               const uint8_t* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint8_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromuint16(double* toptr,
                                                int64_t tooffset,
                                                const uint16_t* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint16_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromuint32(double* toptr,
                                                int64_t tooffset,
                                                const uint32_t* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint32_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromuint64(double* toptr,
                                                int64_t tooffset,
                                                const uint64_t* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<uint64_t, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromfloat32(double* toptr,
                                                 int64_t tooffset,
                                                 const float* fromptr,
                                                 int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<float, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tocomplex128_fromfloat64(double* toptr,
                                                 int64_t tooffset,
                                                 const double* fromptr,
                                                 int64_t length) {
  return awkward_NumpyArray_fill_tocomplex<double, double>(
      toptr, tooffset, fromptr, length);
}
