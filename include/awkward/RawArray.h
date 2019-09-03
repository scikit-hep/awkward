// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RAWARRAY_H_
#define AWKWARD_RAWARRAY_H_

#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T>
  class RawArrayOf: public Content {
  public:
    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<T> ptr, const int64_t offset, const int64_t length, const int64_t stride)
        : id_(id)
        , ptr_(ptr)
        , offset_(offset)
        , length_(length)
        , stride_(stride) {
          assert(sizeof(T) <= stride);
        }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const std::shared_ptr<T> ptr, const int64_t length)
        : id_(id)
        , ptr_(ptr)
        , offset_(0)
        , length_(length)
        , stride_(sizeof(T)) { }

    RawArrayOf<T>(const std::shared_ptr<Identity> id, const int64_t length)
        : id_(id)
        , ptr_(std::shared_ptr<T>(new T[length], awkward::util::array_deleter<T>()))
        , offset_(0)
        , length_(length)
        , stride_(sizeof(T)) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    const int64_t offset() const { return offset_; }
    const int64_t stride() const { return stride_; }

    bool isempty() const;
    bool iscompact() const;
    ssize_t byteoffset() const;
    void* byteptr() const;
    ssize_t bytelength() const;
    uint8_t getbyte(ssize_t at) const;

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    virtual const int64_t length() const;
    virtual const std::shared_ptr<Content> shallow_copy() const;
    virtual const std::shared_ptr<Content> get(int64_t at) const;
    virtual const std::shared_ptr<Content> slice(int64_t start, int64_t stop) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;

    T* borrow(int64_t at) const;

  private:
    std::shared_ptr<Identity> id_;
    const std::shared_ptr<T> ptr_;
    const int64_t offset_;
    const int64_t length_;
    const int64_t stride_;
  };
}

#endif // AWKWARD_RAWARRAY_H_
