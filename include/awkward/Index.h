// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <string>
#include <memory>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"

namespace awkward {
  class Index {
    virtual const std::shared_ptr<Index> shallow_copy() const = 0;
  };

  template <typename T>
  class IndexOf: public Index {
  public:
    IndexOf<T>(int64_t length)
        : ptr_(std::shared_ptr<T>(length == 0 ? nullptr : new T[(size_t)length], awkward::util::array_deleter<T>()))
        , offset_(0)
        , length_(length) { }
    IndexOf<T>(const std::shared_ptr<T> ptr, int64_t offset, int64_t length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    int64_t offset() const { return offset_; }
    int64_t length() const { return length_; }

    const std::string classname() const;
    const std::string tostring() const;
    const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    T getitem_at(int64_t at) const;
    T getitem_at_unsafe(int64_t at) const;
    IndexOf<T> getitem_range(int64_t start, int64_t stop) const;
    IndexOf<T> getitem_range_unsafe(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Index> shallow_copy() const;

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
