// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UTIL_H_
#define AWKWARD_UTIL_H_

#include <string>
#include <vector>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  class Identity;

  namespace util {
    void handle_error(const struct Error& err, const std::string& classname, const Identity* id);

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

    std::string quote(const std::string& x, bool doublequote);
    bool subset(const std::vector<std::string>& super, const std::vector<std::string>& sub);

    template <typename T>
    Error awkward_identity64_from_listoffsetarray(int64_t* toptr, const int64_t* fromptr, const T* fromoffsets, int64_t fromptroffset, int64_t offsetsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    Error awkward_identity64_from_listarray(int64_t* toptr, const int64_t* fromptr, const T* fromstarts, const T* fromstops, int64_t fromptroffset, int64_t startsoffset, int64_t stopsoffset, int64_t tolength, int64_t fromlength, int64_t fromwidth);
    template <typename T>
    Error awkward_listarray_getitem_next_at_64(int64_t* tocarry, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t at);
    template <typename T>
    Error awkward_listarray_getitem_next_range_carrylength(int64_t* carrylength, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
    template <typename T>
    Error awkward_listarray_getitem_next_range_64(T* tooffsets, int64_t* tocarry, const T* fromstarts, const T* fromstops, int64_t lenstarts, int64_t startsoffset, int64_t stopsoffset, int64_t start, int64_t stop, int64_t step);
    template <typename T>
    Error awkward_listarray_getitem_next_range_counts_64(int64_t* total, const T* fromoffsets, int64_t lenstarts);
    template <typename T>
    Error awkward_listarray_getitem_next_range_spreadadvanced_64(int64_t* toadvanced, const int64_t* fromadvanced, const T* fromoffsets, int64_t lenstarts);
    template <typename T>
    Error awkward_listarray_getitem_next_array_64(int64_t* tocarry, int64_t* toadvanced, const T* fromstarts, const T* fromstops, const int64_t* fromarray, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
    template <typename T>
    Error awkward_listarray_getitem_next_array_advanced_64(int64_t* tocarry, int64_t* toadvanced, const T* fromstarts, const T* fromstops, const int64_t* fromarray, const int64_t* fromadvanced, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lenarray, int64_t lencontent);
    template <typename T>
    Error awkward_listarray_getitem_carry_64(T* tostarts, T* tostops, const T* fromstarts, const T* fromstops, const int64_t* fromcarry, int64_t startsoffset, int64_t stopsoffset, int64_t lenstarts, int64_t lencarry);

  }
}

#endif // AWKWARD_UTIL_H_
