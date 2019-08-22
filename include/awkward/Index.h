// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>
#include <memory>

#include "awkward/util.h"

#include <iostream>

namespace awkward {
  class Index {
  public:
    Index(IndexType length)
        : ptr_(std::shared_ptr<IndexType>(new IndexType[length], awkward::util::array_deleter<IndexType>()))
        , offset_(0)
        , length_(length) { }

    Index(const std::shared_ptr<IndexType> ptr, IndexType offset, IndexType length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    ~Index() {
      std::cout << "Index destructor" << std::endl;
    }

    const std::shared_ptr<IndexType> ptr() const { return ptr_; }
    IndexType offset() const { return offset_; }
    IndexType length() const { return length_; }

    const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    IndexType get(AtType at) const;
    Index slice(AtType start, AtType stop) const;

  private:
    const std::shared_ptr<IndexType> ptr_;
    const IndexType offset_;
    const IndexType length_;
  };
}

#endif // AWKWARD_INDEX_H_
