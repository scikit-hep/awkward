// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/builder/ArrayBuilderOptions.h"

#include "awkward/builder/GrowableBuffer.h"

namespace awkward {
  template <typename T>
  GrowableBuffer<T>
  GrowableBuffer<T>::empty(const ArrayBuilderOptions& options) {
    return GrowableBuffer<T>::empty(options, 0);
  }

  template <typename T>
  GrowableBuffer<T>
  GrowableBuffer<T>::empty(const ArrayBuilderOptions& options,
                           int64_t minreserve) {
    size_t actual = (size_t)options.initial();
    if (actual < (size_t)minreserve) {
      actual = (size_t)minreserve;
    }
    std::shared_ptr<T> ptr = kernel::malloc<T>(kernel::lib::cpu, (int64_t)(actual*sizeof(T)));
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
    std::shared_ptr<T> ptr = kernel::malloc<T>(kernel::lib::cpu, (int64_t)(actual*sizeof(T)));
    T* rawptr = ptr.get();
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
                       kernel::malloc<T>(kernel::lib::cpu, options.initial()*(int64_t)sizeof(T)),
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
      std::shared_ptr<T> ptr = kernel::malloc<T>(kernel::lib::cpu, minreserved*(int64_t)sizeof(T));
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
    ptr_ = kernel::malloc<T>(kernel::lib::cpu, options_.initial()*(int64_t)sizeof(T));
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

  template class EXPORT_TEMPLATE_INST GrowableBuffer<bool>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<int8_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<int16_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<int32_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<int64_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<uint8_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<uint16_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<uint32_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<uint64_t>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<float>;
  template class EXPORT_TEMPLATE_INST GrowableBuffer<double>;
}
