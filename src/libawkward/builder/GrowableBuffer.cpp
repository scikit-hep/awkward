// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "awkward/builder/GrowableBuffer.h"

namespace awkward {
  template <typename T>
  GrowableBuffer<T>
  GrowableBuffer<T>::empty(const ArrayBuilderOptions& options) {
    return
  GrowableBuffer<T>::empty(options, 0);
  }

  template <typename T>
  GrowableBuffer<T>
  GrowableBuffer<T>::empty(const ArrayBuilderOptions& options,
                           int64_t minreserve) {
    size_t actual = (size_t)options.initial();
    if (actual < (size_t)minreserve) {
      actual = (size_t)minreserve;
    }
    std::shared_ptr<T> ptr(new T[actual], kernel::array_deleter<T>());
    return GrowableBuffer(options, ptr, 0, (int64_t)actual);
  }

  template <typename T>
  GrowableBuffer<T>
  GrowableBuffer<T>::full(const ArrayBuilderOptions& options,
                          T value,
                          int64_t length) {
    GrowableBuffer<T> out = empty(options, length);
    T* rawptr = out.ptr().get();
    for (int64_t i = 0;  i < length;  i++) {
      rawptr[i] = value;
    }
    return GrowableBuffer<T>(options, out.ptr(), length, out.reserved());
  }

  template <typename T>
  GrowableBuffer<T>
  GrowableBuffer<T>::arange(const ArrayBuilderOptions& options,
                            int64_t length) {
    size_t actual = (size_t)options.initial();
    if (actual < (size_t)length) {
      actual = (size_t)length;
    }
    T* rawptr = new T[(size_t)actual];
    std::shared_ptr<T> ptr(rawptr, kernel::array_deleter<T>());
    for (int64_t i = 0;  i < length;  i++) {
      rawptr[i] = (T)i;
    }
    return GrowableBuffer(options, ptr, length, (int64_t)actual);
  }

  template <typename T>
  GrowableBuffer<T>::GrowableBuffer(const ArrayBuilderOptions& options,
                                    std::shared_ptr<T> ptr,
                                    int64_t length,
                                    int64_t reserved)
      : options_(options)
      , ptr_(ptr)
      , length_(length)
      , reserved_(reserved) { }

  template <typename T>
  GrowableBuffer<T>::GrowableBuffer(const ArrayBuilderOptions& options)
      : GrowableBuffer(options,
                       std::shared_ptr<T>(new T[(size_t)options.initial()],
                                          kernel::array_deleter<T>()),
                       0,
                       options.initial()) { }

  template <typename T>
  const std::shared_ptr<T>
  GrowableBuffer<T>::ptr() const {
    return ptr_;
  }

  template <typename T>
  int64_t
  GrowableBuffer<T>::length() const {
    return length_;
  }

  template <typename T>
  void
  GrowableBuffer<T>::set_length(int64_t newlength) {
    if (newlength > reserved_) {
      set_reserved(newlength);
    }
    length_ = newlength;
  }

  template <typename T>
  int64_t
  GrowableBuffer<T>::reserved() const {
    return reserved_;
  }

  template <typename T>
  void
  GrowableBuffer<T>::set_reserved(int64_t minreserved) {
    if (minreserved > reserved_) {
      std::shared_ptr<T> ptr(new T[(size_t)minreserved],
                             kernel::array_deleter<T>());
      memcpy(ptr.get(), ptr_.get(), (size_t)length_ * sizeof(T));
      ptr_ = ptr;
      reserved_ = minreserved;
    }
  }

  template <typename T>
  void
  GrowableBuffer<T>::clear() {
    length_ = 0;
    reserved_ = options_.initial();
    ptr_ = std::shared_ptr<T>(new T[(size_t)options_.initial()],
                              kernel::array_deleter<T>());
  }

  template <typename T>
  void
  GrowableBuffer<T>::append(T datum) {
    if (length_ == reserved_) {
      set_reserved((int64_t)ceil(reserved_ * options_.resize()));
    }
    ptr_.get()[length_] = datum;
    length_++;
  }

  template <typename T>
  T
  GrowableBuffer<T>::getitem_at_nowrap(int64_t at) const {
    return ptr_.get()[at];
  }

  template class EXPORT_TEMPLATE_INST GrowableBuffer<int8_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<uint8_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<int64_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<double>;
}
