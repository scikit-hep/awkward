// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UTIL_H_
#define AWKWARD_UTIL_H_

#include <iostream>

#ifdef _MSC_VER
#include <BaseTsd.h>
#define ssize_t SSIZE_T
#else
#include <cstdint>
#endif

namespace awkward {
  typedef uint32_t AtType;
  typedef int32_t IndexType;
  typedef uint8_t byte;

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
