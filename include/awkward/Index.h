// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <memory>

#include "awkward/util.h"

namespace awkward {
  template <typename T>
  class IndexOf {
  public:
    IndexOf<T>(T length)
        : ptr_(std::shared_ptr<T>(new T[length], awkward::util::array_deleter<T>()))
        , offset_(0)
        , length_(length) { }
    IndexOf<T>(const std::shared_ptr<T> ptr, T offset, T length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    T offset() const { return offset_; }
    T length() const { return length_; }

    const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    T get(IndexType at) const;
    IndexOf<T> slice(IndexType start, IndexType stop) const;

  private:
    const std::shared_ptr<T> ptr_;
    const T offset_;
    const T length_;
  };

  typedef IndexOf<IndexType> Index;
  typedef IndexOf<TagType> TagIndex;
  typedef IndexOf<ChunkOffsetType> ChunkOffsetIndex;
}

#endif // AWKWARD_INDEX_H_
