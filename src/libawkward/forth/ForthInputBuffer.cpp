// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/forth/ForthInputBuffer.h"

namespace awkward {
  ForthInputBuffer::ForthInputBuffer(const std::shared_ptr<void> ptr,
                                     int64_t offset,
                                     int64_t length)
    : ptr_(ptr)
    , offset_(offset)
    , length_(length)
    , pos_(0) { }
}
