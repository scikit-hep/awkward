// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <cmath>
#include <cstring>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  template <typename T>
  class GrowableBuffer {
  public:
    GrowableBuffer(int64_t initial, double resize): ptr_(new T[initial], awkward::util::array_deleter<T>()), length_(0), reserved_(initial), initial_(initial), resize_(resize) { }
    GrowableBuffer(): GrowableBuffer(1024, 2.0) { }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    int64_t length() const { return length_; }
    int64_t reserved() const { return reserved_; }
    int64_t initial() const { return initial_; }
    double resize() const { return resize_; }

    void clear() {
      length_ = 0;
      reserved_ = initial_;
      ptr_ = std::shared_ptr<T>(new T[initial_], awkward::util::array_deleter<T>());
    }

    void append(T datum) {
      if (length_ == reserved_) {
        size_t newsize = (size_t)(ceil(reserved_ * resize_));
        std::shared_ptr<T> ptr(new T[newsize], awkward::util::array_deleter<T>());
        memcpy(ptr.get(), ptr_.get(), (size_t)(length_ * sizeof(T)));
        reserved_ = (int64_t)newsize;
        ptr_ = ptr;
      }
      ptr_.get()[length_] = datum;
      length_++;
    }

  private:
    std::shared_ptr<T> ptr_;
    int64_t length_;
    int64_t reserved_;
    int64_t initial_;
    double resize_;
  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_
