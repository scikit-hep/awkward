// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/kernel-dispatch.h"

#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  ////////// abstract

  ForthOutputBuffer::ForthOutputBuffer(int64_t initial, double resize)
    : length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  ////////// specialized

  template <typename OUT>
  ForthOutputBufferOf<OUT>::ForthOutputBufferOf(int64_t initial, double resize)
    : ForthOutputBuffer(initial, resize)
    , ptr_(new OUT[initial], kernel::array_deleter<OUT>()) { }

  template <typename OUT>
  const std::shared_ptr<void>
  ForthOutputBufferOf<OUT>::ptr() const noexcept {
    return ptr_;
  }

  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<bool>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<int8_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<int16_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<int32_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<int64_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<uint8_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<uint16_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<uint32_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<uint64_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<float>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<double>;

}
