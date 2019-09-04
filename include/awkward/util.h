// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UTIL_H_
#define AWKWARD_UTIL_H_

#include "awkward/cpu-kernels/util.h"

#ifdef _MSC_VER
  #define PYBIND11_INT32_FORMAT "l"
  #define PYBIND11_INT64_FORMAT "q"
#else
  #define PYBIND11_INT32_FORMAT "i"
  #define PYBIND11_INT64_FORMAT "l"
#endif

namespace awkward {
  #define HANDLE_ERROR(err) { if (err != kNoError) { throw std::invalid_argument(err); } }

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
