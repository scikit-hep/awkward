// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill_fromcomplex.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill_fromcomplex(TO* toptr,
                                    int64_t tooffset,
                                    const FROM* fromptr,
                                    int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = (TO)fromptr[i*2];
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_tobool_fromcomplex64(bool* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
 return awkward_NumpyArray_fill_fromcomplex<float, bool>(
     toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint8_fromcomplex64(int8_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromcomplex64(int16_t* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromcomplex64(int32_t* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromcomplex64(int64_t* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromcomplex64(uint8_t* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromcomplex64(uint16_t* toptr,
                                               int64_t tooffset,
                                               const float* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromcomplex64(uint32_t* toptr,
                                               int64_t tooffset,
                                               const float* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromcomplex64(uint64_t* toptr,
                                               int64_t tooffset,
                                               const float* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromcomplex64(float* toptr,
                                                int64_t tooffset,
                                                const float* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromcomplex64(double* toptr,
                                                int64_t tooffset,
                                                const float* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<float, double>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tobool_fromcomplex128(bool* toptr,
                                              int64_t tooffset,
                                              const double* fromptr,
                                              int64_t length) {
 return awkward_NumpyArray_fill_fromcomplex<double, bool>(
     toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint8_fromcomplex128(int8_t* toptr,
                                              int64_t tooffset,
                                              const double* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromcomplex128(int16_t* toptr,
                                               int64_t tooffset,
                                               const double* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromcomplex128(int32_t* toptr,
                                               int64_t tooffset,
                                               const double* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromcomplex128(int64_t* toptr,
                                               int64_t tooffset,
                                               const double* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromcomplex128(uint8_t* toptr,
                                               int64_t tooffset,
                                               const double* fromptr,
                                               int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromcomplex128(uint16_t* toptr,
                                                int64_t tooffset,
                                                const double* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromcomplex128(uint32_t* toptr,
                                                int64_t tooffset,
                                                const double* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromcomplex128(uint64_t* toptr,
                                                int64_t tooffset,
                                                const double* fromptr,
                                                int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromcomplex128(float* toptr,
                                                 int64_t tooffset,
                                                 const double* fromptr,
                                                 int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromcomplex128(double* toptr,
                                                 int64_t tooffset,
                                                 const double* fromptr,
                                                 int64_t length) {
  return awkward_NumpyArray_fill_fromcomplex<double, double>(
      toptr, tooffset, fromptr, length);
}
