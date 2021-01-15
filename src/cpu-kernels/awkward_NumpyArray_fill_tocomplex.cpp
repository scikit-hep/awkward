// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill_tocomplex.cpp", line)

#include <complex>

#include "awkward/kernels.h"

// For any object z of type complex<T>,
// reinterpret_cast<T(&)[2]>(z)[0] is the real part of z and
// reinterpret_cast<T(&)[2]>(z)[1] is the imaginary part of z.
// For any pointer to an element of an array of complex<T> named p and
// any valid array index i,
// reinterpret_cast<T*>(p)[2*i] is the real part of the complex number p[i], and
// reinterpret_cast<T*>(p)[2*i + 1] is the imaginary part of the complex number p[i]
// The intent of this requirement is to preserve binary compatibility between
// the C++ library complex number types and the C language complex number types
// (and arrays thereof), which have an identical object representation requirement.

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill_tocomplex(std::complex<TO>* toptr,
                                  int64_t tooffset,
                                  const std::complex<FROM>* fromptr,
                                  int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = {(TO)fromptr[i].real(), (TO) fromptr[i].imag()};
  }
  return success();
}
template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill_tocomplex(TO* toptr,
                                  int64_t tooffset,
                                  const FROM* fromptr,
                                  int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + 2*i] = (TO)fromptr[i];
    toptr[tooffset + 2*i + 1] = (TO)0;
  }
  return success();
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
awkward_NumpyArray_fill_tocomplex64_fromcomplex128(float* toptr,
                                                   int64_t tooffset,
                                                   const double* fromptr,
                                                   int64_t length) {
  return awkward_NumpyArray_fill<double, float>(
    toptr, tooffset, fromptr, length);
  return success();
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
