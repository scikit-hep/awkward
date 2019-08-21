// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <string>
#include <sstream>
#include <memory>

#include "awkward/util.h"

namespace awkward {
  class Index {
  public:
    Index(IndexType length)
        : ptr_(std::shared_ptr<IndexType>(new IndexType[length], awkward::util::array_deleter<IndexType>()))
        , offset_(0)
        , length_(length) { }

    Index(std::shared_ptr<IndexType> ptr, IndexType offset, IndexType length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    std::shared_ptr<IndexType> ptr() const { return ptr_; }
    IndexType offset() const { return offset_; }
    IndexType length() const { return length_; }

    const std::string repr() const;
    IndexType get(IndexType at) const; // FIXME: AtType
    Index slice(IndexType start, IndexType stop) const; // FIXME: AtType, AtType

  private:
    const std::shared_ptr<IndexType> ptr_;
    const IndexType offset_;
    const IndexType length_;
  };
}

#endif // AWKWARD_INDEX_H_
