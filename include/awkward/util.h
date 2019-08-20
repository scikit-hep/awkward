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
  namespace util {

    // https://stackoverflow.com/a/13062069/1623645
    template<typename T>
    struct array_deleter {
      void operator()(T const *p) {
        delete[] p;
      }
    };

    template<typename T>
    struct no_deleter {
      void operator()(T const *p) { }
    };

  }
}

#endif // AWKWARD_UTIL_H_
