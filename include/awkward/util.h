// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UTIL_H_
#define AWKWARD_UTIL_H_

#include <iostream>

#ifdef _MSC_VER
  #ifdef _WIN64
    typedef signed __int64 ssize_t;
  #else
    typedef signed   int   ssize_t;
  #endif
  typedef unsigned char  uint8_t;
  typedef signed   int   int32_t;
  typedef signed __int64 int64_t;
#else
  #include <cstdint>
#endif

namespace awkward {
  typedef int64_t AtType;
  typedef int32_t IndexType;
  typedef uint8_t byte;

  const IndexType MAXSIZE = 2147483647;   // 2**31 - 1

  namespace util {

    template<typename T>
    class array_deleter {
    public:
      void operator()(T const *p) {
        delete[] p;
      }
    };

    template<typename T>
    class no_deleter {
    public:
      void operator()(T const *p) { }
    };

  }
}

#endif // AWKWARD_UTIL_H_
