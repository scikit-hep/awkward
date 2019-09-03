// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <memory>
#include <type_traits>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  template <typename T>
  class IndexOf {
  public:
    IndexOf<T>(T length)
        : ptr_(std::shared_ptr<T>(new T[length], awkward::util::array_deleter<T>()))
        , offset_(0)
        , length_(length) { }
    IndexOf<T>(const std::shared_ptr<T> ptr, int64_t offset, int64_t length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    int64_t offset() const { return offset_; }
    int64_t length() const { return length_; }

    const std::string tostring() const;
    const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    T get(int64_t at) const;
    IndexOf<T> slice(int64_t start, int64_t stop) const;

  private:
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
  };

  typedef IndexOf<uint8_t> Index8;
  typedef IndexOf<int32_t> Index32;
  typedef IndexOf<int64_t> Index64;
}

#endif // AWKWARD_INDEX_H_
