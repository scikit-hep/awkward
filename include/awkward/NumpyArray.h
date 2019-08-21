// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_NUMPYARRAYINDEX_H_
#define AWKWARD_NUMPYARRAYINDEX_H_

#include <cassert>
#include <vector>

#include "awkward/util.h"

namespace awkward {
  class NumpyArray {
  public:
    NumpyArray(std::shared_ptr<byte> ptr, const std::vector<ssize_t> shape, const std::vector<ssize_t> strides, ssize_t bytepos, ssize_t itemsize, const std::string format)
        : ptr_(ptr)
        , shape_(shape)
        , strides_(strides)
        , bytepos_(bytepos)
        , itemsize_(itemsize)
        , format_(format) {
          assert(shape_.size() == strides_.size());
        }

    const std::shared_ptr<byte> ptr() const { return ptr_; }
    const std::vector<ssize_t> shape() const { return shape_; }
    const std::vector<ssize_t> strides() const { return strides_; }
    ssize_t bytepos() const { return bytepos_; }
    ssize_t itemsize() const { return itemsize_; }
    const std::string format() const { return format_; }

    IndexType ndim() const { return shape_.size(); }
    bool isscalar() const { return ndim() == 0; }
    bool isempty() const {
      for (auto x : shape_) {
        if (x == 0) return true;
      }
      return false;  // false for isscalar(), too
    }
    void* byteptr() const { return reinterpret_cast<void*>(reinterpret_cast<ssize_t>(ptr_.get()) + bytepos_); }

    NumpyArray get(IndexType at) const { // FIXME: AtType
      assert(!isscalar());
      assert(0 <= at  &&  at < shape_[shape_.size() - 1]);
      const std::vector<ssize_t> shape(shape_.begin() + 1, shape_.end());
      const std::vector<ssize_t> strides(strides_.begin() + 1, strides_.end());
      ssize_t bytepos = bytepos_ + strides_[0]*at;
      return NumpyArray(ptr_, shape, strides, bytepos, itemsize_, format_);
    }

  private:
    const std::shared_ptr<byte> ptr_;
    const std::vector<ssize_t> shape_;
    const std::vector<ssize_t> strides_;
    const ssize_t bytepos_;
    const ssize_t itemsize_;
    const std::string format_;
  };
}

#endif // AWKWARD_NUMPYARRAYINDEX_H_
