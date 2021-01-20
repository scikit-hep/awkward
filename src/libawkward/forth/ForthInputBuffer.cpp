// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthInputBuffer.cpp", line)

#include "awkward/forth/ForthInputBuffer.h"

namespace awkward {
  ForthInputBuffer::ForthInputBuffer(const std::shared_ptr<void> ptr,
                                     int64_t offset,
                                     int64_t length)
    : ptr_(ptr)
    , offset_(offset)
    , length_(length)
    , pos_(0) { }

  void*
  ForthInputBuffer::read(int64_t num_bytes, util::ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next > length_) {
      err = util::ForthError::read_beyond;
      return nullptr;
    }
    void* out = reinterpret_cast<void*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)pos_
    );
    pos_ = next;
    return out;
  }

  void
  ForthInputBuffer::seek(int64_t to, util::ForthError& err) noexcept {
    if (to < 0  ||  to > length_) {
      err = util::ForthError::seek_beyond;
    }
    else {
      pos_ = to;
    }
  }

  void
  ForthInputBuffer::skip(int64_t num_bytes, util::ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next < 0  ||  next > length_) {
      err = util::ForthError::skip_beyond;
    }
    else {
      pos_ = next;
    }
  }

  bool
  ForthInputBuffer::end() const noexcept {
    return pos_ == length_;
  }

  int64_t
  ForthInputBuffer::pos() const noexcept {
    return pos_;
  }

  int64_t
  ForthInputBuffer::len() const noexcept {
    return length_;
  }

}
