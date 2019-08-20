// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEX_H_
#define AWKWARD_INDEX_H_

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <string>
#include <sstream>
#include <memory>

#include "util.h"

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

    Index(IndexType *ptr, IndexType offset, IndexType length)
        : ptr_(std::shared_ptr<IndexType>(ptr, awkward::util::no_deleter<IndexType>()))
        , offset_(offset)
        , length_(length) { }

    Index(std::shared_ptr<IndexType> ptr, IndexType offset, IndexType length)
        : ptr_(ptr)
        , offset_(offset)
        , length_(length) { }

    std::string repr() {
      std::stringstream out;
      out << "<Index [";
      if (len() <= 10) {
        for (int i = 0;  i < len();  i++) {
          if (i != 0) {
            out << ", ";
          }
          out << get(i);
        }
      }
      else {
        for (int i = 0;  i < 5;  i++) {
          if (i != 0) {
            out << ", ";
          }
          out << get(i);
        }
        out << " ... ";
        for (int i = len() - 6;  i < len();  i++) {
          if (i != len() - 6) {
            out << ", ";
          }
          out << get(i);
        }
      }
      out << "] at 0x";
      out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << ">";
      return out.str();
    }

    IndexType len() {
      return length_;
    }

    IndexType get(IndexType at) {
      assert(0 <= at  &&  at < length_);
      return ptr_.get()[offset_ + at];
    }

    Index slice(IndexType start, IndexType stop) {
      assert(start == stop  ||  (0 <= start  &&  start < length_));
      assert(start == stop  ||  (0 < stop    &&  stop <= length_));
      assert(start <= stop);
      return Index(ptr_, offset_ + start*(start != stop), stop - start);
    }

  private:
    std::shared_ptr<IndexType> ptr_;   // 16 bytes
    IndexType offset_;                 //  4 bytes
    IndexType length_;                 //  4 bytes
  };
}

#endif // AWKWARD_INDEX_H_
