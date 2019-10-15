// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <cmath>
#include <cstring>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"

namespace awkward {
  template <typename T>
  class GrowableBuffer {
  public:
    GrowableBuffer(const FillableOptions& options, std::shared_ptr<T> ptr, int64_t length, int64_t reserved): options_(options), ptr_(ptr), length_(length), reserved_(reserved) { }
    GrowableBuffer(const FillableOptions& options): GrowableBuffer(options, std::shared_ptr<T>(new T[(size_t)options.initial()], awkward::util::array_deleter<T>()), 0, options.initial()) { }

    static GrowableBuffer<T> full(const FillableOptions& options, T value, int64_t length) {
      size_t actual = (size_t)options.initial();
      if (actual < (size_t)length) {
        actual = (size_t)length;
      }
      T* rawptr = new T[(size_t)actual];
      std::shared_ptr<T> ptr(rawptr, awkward::util::array_deleter<T>());
      for (int64_t i = 0;  i < length;  i++) {
        rawptr[i] = value;
      }
      return GrowableBuffer(options, ptr, length, (int64_t)actual);
    }

    static GrowableBuffer<T> arange(const FillableOptions& options, int64_t length) {
      size_t actual = (size_t)options.initial();
      if (actual < (size_t)length) {
        actual = (size_t)length;
      }
      T* rawptr = new T[(size_t)actual];
      std::shared_ptr<T> ptr(rawptr, awkward::util::array_deleter<T>());
      for (int64_t i = 0;  i < length;  i++) {
        rawptr[i] = (T)i;
      }
      return GrowableBuffer(options, ptr, length, (int64_t)actual);
    }

    const std::shared_ptr<T> ptr() const { return ptr_; }
    int64_t length() const { return length_; }
    int64_t reserved() const { return reserved_; }

    void clear() {
      length_ = 0;
      reserved_ = options_.initial();
      ptr_ = std::shared_ptr<T>(new T[(size_t)options_.initial()], awkward::util::array_deleter<T>());
    }

    void append(T datum) {
      if (length_ == reserved_) {
        size_t newsize = (size_t)(ceil(reserved_ * options_.resize()));
        if (newsize < (size_t)reserved_) {
          newsize = (size_t)(reserved_ + 1);
        }
        std::shared_ptr<T> ptr(new T[newsize], awkward::util::array_deleter<T>());
        memcpy(ptr.get(), ptr_.get(), (size_t)(length_ * sizeof(T)));
        reserved_ = (int64_t)newsize;
        ptr_ = ptr;
      }
      ptr_.get()[length_] = datum;
      length_++;
    }

  private:
    const FillableOptions options_;
    std::shared_ptr<T> ptr_;
    int64_t length_;
    int64_t reserved_;
  };
}

#endif // AWKWARD_GROWABLEBUFFER_H_
