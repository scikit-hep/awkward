// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_NumpyArray_fill.cpp", line)

#include "awkward/kernels.h"

template <typename FROM, typename TO>
ERROR
awkward_NumpyArray_fill(TO* toptr,
                        int64_t tooffset,
                        const FROM* fromptr,
                        int64_t length) {
  for (int64_t i = 0; i < length; i++) {
    toptr[tooffset + i] = (TO)fromptr[i];
  }
  return success();
}
ERROR
awkward_NumpyArray_fill_toint8_fromint8(int8_t* toptr,
                                        int64_t tooffset,
                                        const int8_t* fromptr,
                                        int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint8(int16_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint8(int32_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint8(int64_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint8(uint8_t* toptr,
                                         int64_t tooffset,
                                         const int8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint8(uint16_t* toptr,
                                          int64_t tooffset,
                                          const int8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint8(uint32_t* toptr,
                                          int64_t tooffset,
                                          const int8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint8(uint64_t* toptr,
                                          int64_t tooffset,
                                          const int8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int8_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint8(float* toptr,
                                           int64_t tooffset,
                                           const int8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int8_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint8(double* toptr,
                                           int64_t tooffset,
                                           const int8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int8_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromint16(int8_t* toptr,
                                         int64_t tooffset,
                                         const int16_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint16(int16_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint16(int32_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint16(int64_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint16(uint8_t* toptr,
                                          int64_t tooffset,
                                          const int16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint16(uint16_t* toptr,
                                           int64_t tooffset,
                                           const int16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint16(uint32_t* toptr,
                                           int64_t tooffset,
                                           const int16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint16(uint64_t* toptr,
                                           int64_t tooffset,
                                           const int16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int16_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint16(float* toptr,
                                            int64_t tooffset,
                                            const int16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int16_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint16(double* toptr,
                                            int64_t tooffset,
                                            const int16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int16_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromint32(int8_t* toptr,
                                         int64_t tooffset,
                                         const int32_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint32(int16_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint32(int32_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint32(int64_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint32(uint8_t* toptr,
                                          int64_t tooffset,
                                          const int32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint32(uint16_t* toptr,
                                           int64_t tooffset,
                                           const int32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint32(uint32_t* toptr,
                                           int64_t tooffset,
                                           const int32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint32(uint64_t* toptr,
                                           int64_t tooffset,
                                           const int32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int32_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint32(float* toptr,
                                            int64_t tooffset,
                                            const int32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int32_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint32(double* toptr,
                                            int64_t tooffset,
                                            const int32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int32_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromint64(int8_t* toptr,
                                         int64_t tooffset,
                                         const int64_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromint64(int16_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromint64(int32_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromint64(int64_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromint64(uint8_t* toptr,
                                          int64_t tooffset,
                                          const int64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromint64(uint16_t* toptr,
                                           int64_t tooffset,
                                           const int64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromint64(uint32_t* toptr,
                                           int64_t tooffset,
                                           const int64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromint64(uint64_t* toptr,
                                           int64_t tooffset,
                                           const int64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<int64_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromint64(float* toptr,
                                            int64_t tooffset,
                                            const int64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int64_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromint64(double* toptr,
                                            int64_t tooffset,
                                            const int64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<int64_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint8(int8_t* toptr,
                                         int64_t tooffset,
                                         const uint8_t* fromptr,
                                         int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint8(int16_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint8(int32_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint8(int64_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint8(uint8_t* toptr,
                                          int64_t tooffset,
                                          const uint8_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint8(uint16_t* toptr,
                                           int64_t tooffset,
                                           const uint8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint8(uint32_t* toptr,
                                           int64_t tooffset,
                                           const uint8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint8(uint64_t* toptr,
                                           int64_t tooffset,
                                           const uint8_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint8(float* toptr,
                                            int64_t tooffset,
                                            const uint8_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint8(double* toptr,
                                            int64_t tooffset,
                                            const uint8_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint8_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint16(int8_t* toptr,
                                          int64_t tooffset,
                                          const uint16_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint16(int16_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint16(int32_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint16(int64_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint16(uint8_t* toptr,
                                           int64_t tooffset,
                                           const uint16_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint16(uint16_t* toptr,
                                            int64_t tooffset,
                                            const uint16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint16(uint32_t* toptr,
                                            int64_t tooffset,
                                            const uint16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint16(uint64_t* toptr,
                                            int64_t tooffset,
                                            const uint16_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint16(float* toptr,
                                             int64_t tooffset,
                                             const uint16_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint16(double* toptr,
                                             int64_t tooffset,
                                             const uint16_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint16_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint32(int8_t* toptr,
                                          int64_t tooffset,
                                          const uint32_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint32(int16_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint32(int32_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint32(int64_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint32(uint8_t* toptr,
                                           int64_t tooffset,
                                           const uint32_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint32(uint16_t* toptr,
                                            int64_t tooffset,
                                            const uint32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint32(uint32_t* toptr,
                                            int64_t tooffset,
                                            const uint32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint32(uint64_t* toptr,
                                            int64_t tooffset,
                                            const uint32_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint32(float* toptr,
                                             int64_t tooffset,
                                             const uint32_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint32(double* toptr,
                                             int64_t tooffset,
                                             const uint32_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint32_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromuint64(int8_t* toptr,
                                          int64_t tooffset,
                                          const uint64_t* fromptr,
                                          int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromuint64(int16_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromuint64(int32_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromuint64(int64_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromuint64(uint8_t* toptr,
                                           int64_t tooffset,
                                           const uint64_t* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromuint64(uint16_t* toptr,
                                            int64_t tooffset,
                                            const uint64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromuint64(uint32_t* toptr,
                                            int64_t tooffset,
                                            const uint64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromuint64(uint64_t* toptr,
                                            int64_t tooffset,
                                            const uint64_t* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromuint64(float* toptr,
                                             int64_t tooffset,
                                             const uint64_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromuint64(double* toptr,
                                             int64_t tooffset,
                                             const uint64_t* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<uint64_t, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromfloat32(int8_t* toptr,
                                           int64_t tooffset,
                                           const float* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<float, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromfloat32(int16_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromfloat32(int32_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromfloat32(int64_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromfloat32(uint8_t* toptr,
                                            int64_t tooffset,
                                            const float* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<float, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromfloat32(uint16_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<float, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromfloat32(uint32_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<float, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromfloat32(uint64_t* toptr,
                                             int64_t tooffset,
                                             const float* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<float, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromfloat32(float* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<float, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromfloat32(double* toptr,
                                              int64_t tooffset,
                                              const float* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<float, double>(
      toptr, tooffset, fromptr, length);
}

ERROR
awkward_NumpyArray_fill_toint8_fromfloat64(int8_t* toptr,
                                           int64_t tooffset,
                                           const double* fromptr,
                                           int64_t length) {
  return awkward_NumpyArray_fill<double, int8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint16_fromfloat64(int16_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, int16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint32_fromfloat64(int32_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, int32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_toint64_fromfloat64(int64_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, int64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint8_fromfloat64(uint8_t* toptr,
                                            int64_t tooffset,
                                            const double* fromptr,
                                            int64_t length) {
  return awkward_NumpyArray_fill<double, uint8_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint16_fromfloat64(uint16_t* toptr,
                                             int64_t tooffset,
                                             const double* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<double, uint16_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint32_fromfloat64(uint32_t* toptr,
                                             int64_t tooffset,
                                             const double* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<double, uint32_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_touint64_fromfloat64(uint64_t* toptr,
                                             int64_t tooffset,
                                             const double* fromptr,
                                             int64_t length) {
  return awkward_NumpyArray_fill<double, uint64_t>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat32_fromfloat64(float* toptr,
                                              int64_t tooffset,
                                              const double* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<double, float>(
      toptr, tooffset, fromptr, length);
}
ERROR
awkward_NumpyArray_fill_tofloat64_fromfloat64(double* toptr,
                                              int64_t tooffset,
                                              const double* fromptr,
                                              int64_t length) {
  return awkward_NumpyArray_fill<double, double>(
      toptr, tooffset, fromptr, length);
}
