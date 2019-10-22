// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_GROWABLEBUFFER_H_
#define AWKWARD_GROWABLEBUFFER_H_

#include <cmath>
#include <cstring>
#include <cassert>

#include "awkward/cpu-kernels/util.h"
#include "awkward/fillable/FillableOptions.h"

namespace awkward {
  template <typename T>
  class GrowableBuffer {
  public:
    GrowableBuffer(const FillableOptions& options, std::shared_ptr<T> ptr, int64_t length, int64_t reserved): options_(options), ptr_(ptr), length_(length), reserved_(reserved) { }
    GrowableBuffer(const FillableOptions& options): GrowableBuffer(options, std::shared_ptr<T>(new T[(size_t)options.initial()], awkward::util::array_deleter<T>()), 0, options.initial()) { }

    static GrowableBuffer<T> empty(const FillableOptions& options) {
      return GrowableBuffer<T>::empty(options, 0);
    }

    static GrowableBuffer<T> empty(const FillableOptions& options, int64_t minreserve) {
      size_t actual = (size_t)options.initial();
      if (actual < (size_t)minreserve) {
        actual = (size_t)minreserve;
      }
      std::shared_ptr<T> ptr(new T[actual], awkward::util::array_deleter<T>());
      return GrowableBuffer(options, ptr, 0, (int64_t)actual);
    }

    static GrowableBuffer<T> full(const FillableOptions& options, T value, int64_t length) {
      GrowableBuffer<T> out = empty(options, length);
      T* rawptr = out.ptr().get();
      for (int64_t i = 0;  i < length;  i++) {
        rawptr[i] = value;
      }
      return GrowableBuffer<T>(options, out.ptr(), length, out.reserved());
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
    void set_length(int64_t newlength) {
      if (newlength > reserved_) {
        set_reserved(newlength);
      }
      length_ = newlength;
    }

    int64_t reserved() const { return reserved_; }
    void set_reserved(int64_t minreserved) {
      if (minreserved > reserved_) {
        std::shared_ptr<T> ptr(new T[(size_t)minreserved], awkward::util::array_deleter<T>());
        memcpy(ptr.get(), ptr_.get(), (size_t)(length_ * sizeof(T)));
        ptr_ = ptr;
        reserved_ = minreserved;
      }
    }

    void clear() {
      length_ = 0;
      reserved_ = options_.initial();
      ptr_ = std::shared_ptr<T>(new T[(size_t)options_.initial()], awkward::util::array_deleter<T>());
    }

    void append(T datum) {
      assert(length_ <= reserved_);
      if (length_ == reserved_) {
        set_reserved((int64_t)ceil(reserved_ * options_.resize()));
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
