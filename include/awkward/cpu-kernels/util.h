// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDCPU_UTIL_H_
#define AWKWARDCPU_UTIL_H_

#ifdef _MSC_VER
  #ifdef _WIN64
    typedef signed   __int64 ssize_t;
  #else
    typedef signed   int     ssize_t;
  #endif
  typedef   unsigned char    uint8_t;
  typedef   signed   char    int8_t;
  typedef   signed   int     int32_t;
  typedef   signed   __int64 int64_t;
#else
  #include <cstdint>
#endif

extern "C" {
  typedef const char* Error;
  const Error kNoError = nullptr;

  const int8_t  kMaxInt8  =                 127;   // 2**7  - 1
  const uint8_t kMaxUInt8 =                 255;   // 2**8  - 1
  const int32_t kMaxInt32 =          2147483647;   // 2**31 - 1
  const int64_t kMaxInt64 = 9223372036854775806;   // 2**63 - 2: kMaxInt64 + 1 is Slice::none()
}

#endif // AWKWARDCPU_UTIL_H_
