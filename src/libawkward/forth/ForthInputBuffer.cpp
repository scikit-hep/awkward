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
  ForthInputBuffer::read(int64_t num_bytes, util::ForthError& err) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  void
  ForthInputBuffer::seek(int64_t to, util::ForthError& err) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  void
  ForthInputBuffer::skip(int64_t num_bytes, util::ForthError& err) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  bool
  ForthInputBuffer::end() const {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  int64_t
  ForthInputBuffer::pos() const {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  int64_t
  ForthInputBuffer::len() const {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

}
