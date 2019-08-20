// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RAW_H_
#define AWKWARD_RAW_H_

#include "awkward/Index.h"

namespace awkward {
  typedef uint8_t RawType;

  class Raw {
  public:
    Raw(IndexType entrysize, IndexType length)
        : ptr_(std::shared_ptr<RawType>(new RawType[length], awkward::util::array_deleter<RawType>()))
        , entrysize_(entrysize)
        , offset_(0)
        , length_(length) { }

    Raw(std::shared_ptr<RawType> ptr, IndexType entrysize, IndexType offset, IndexType length)
        : ptr_(ptr)
        , entrysize_(entrysize)
        , offset_(offset)
        , length_(length) { }

    IndexType len() {
      return length_;
    }

    std::string repr();
    RawType getraw(IndexType at);
    Raw slice(IndexType start, IndexType stop);

  private:
    std::shared_ptr<RawType> ptr_;   // 16 bytes
    IndexType entrysize_;            //  4 bytes
    IndexType offset_;               //  4 bytes
    IndexType length_;               //  4 bytes
  };
}

#endif // AWKWARD_RAW_H_
