// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_UTIL_H_
#define AWKWARDCPU_UTIL_H_

#ifdef _MSC_VER
  #ifdef _WIN64
    typedef signed   __int64 ssize_t;
    typedef unsigned __int64 size_t;
  #else
    typedef signed   int     ssize_t;
    typedef unsigned int     size_t;
  #endif
  typedef   unsigned char    uint8_t;
  typedef   signed   char    int8_t;
  typedef   unsigned short   uint16_t;
  typedef   signed   short   int16_t;
  typedef   unsigned int     uint32_t;
  typedef   signed   int     int32_t;
  typedef   unsigned __int64 uint64_t;
  typedef   signed   __int64 int64_t;
#else
  #include <cstddef>
  #include <cstdint>
#endif

#include <iostream>

extern "C" {
  struct Error {
    const char* str;
    int64_t location;
    int64_t attempt;
    int64_t extra;
  };
  struct Error success();
  struct Error failure(const char* str, int64_t location, int64_t attempt);

  const int8_t  kMaxInt8   =                 127;   // 2**7  - 1
  const uint8_t kMaxUInt8  =                 255;   // 2**8  - 1
  const int32_t kMaxInt32  =          2147483647;   // 2**31 - 1
  const int64_t kMaxInt64  = 9223372036854775806;   // 2**63 - 2: see below
  const int64_t kSliceNone = kMaxInt64 + 1;         // reserved for Slice::none()
}

#endif // AWKWARDCPU_UTIL_H_
