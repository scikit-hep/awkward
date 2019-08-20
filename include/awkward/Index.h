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
  typedef int32_t IndexType;

  class Index {
  public:
    Index(IndexType length)
        : ptr_(std::shared_ptr<IndexType>(new IndexType[length], awkward::util::array_deleter<IndexType>()))
        , offset_(0)
        , length_(length) { }

    Index(IndexType *ptr, IndexType length)
        : ptr_(std::shared_ptr<IndexType>(ptr, awkward::util::no_deleter<IndexType>()))
        , offset_(0)
        , length_(length) { }

    Index(std::shared_ptr<IndexType> ptr, IndexType offset, IndexType length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    IndexType len() {
      return length_;
    }

    std::string repr();
    IndexType get(IndexType at);
    Index slice(IndexType start, IndexType stop);

  private:
    std::shared_ptr<IndexType> ptr_;   // 16 bytes
    IndexType offset_;                 //  4 bytes
    IndexType length_;                 //  4 bytes
  };
}

#endif // AWKWARD_INDEX_H_
