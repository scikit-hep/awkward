// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthOutputBuffer.cpp", line)

#include "awkward/kernel-dispatch.h"

#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  ////////// abstract

  ForthOutputBuffer::ForthOutputBuffer(int64_t initial, double resize)
    : length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  ForthOutputBuffer::~ForthOutputBuffer() = default;

  int64_t
  ForthOutputBuffer::len() const {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  void
  ForthOutputBuffer::rewind(int64_t num_bytes, util::ForthError& err) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

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

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_bool(bool value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int8(int8_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int16(int16_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int32(int32_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int64(int64_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_intp(ssize_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint8(uint8_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint16(uint16_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint32(uint32_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint64(uint64_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uintp(size_t value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_float32(float value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_float64(double value, bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_bool(int64_t num_items,
                                       bool* values,
                                       bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int8(int64_t num_items,
                                       int8_t* values,
                                       bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int16(int64_t num_items,
                                        int16_t* values,
                                        bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int32(int64_t num_items,
                                        int32_t* values,
                                        bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int64(int64_t num_items,
                                        int64_t* values,
                                        bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_intp(int64_t num_items,
                                       ssize_t* values,
                                       bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint8(int64_t num_items,
                                        uint8_t* values,
                                        bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint16(int64_t num_items,
                                         uint16_t* values,
                                         bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint32(int64_t num_items,
                                         uint32_t* values,
                                         bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint64(int64_t num_items,
                                         uint64_t* values,
                                         bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uintp(int64_t num_items,
                                        size_t* values,
                                        bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float32(int64_t num_items,
                                          float* values,
                                          bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float64(int64_t num_items,
                                          double* values,
                                          bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<bool>::write_bool(int64_t num_items,
                                        bool* values,
                                        bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<int8_t>::write_int8(int64_t num_items,
                                          int8_t* values,
                                          bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<int16_t>::write_int16(int64_t num_items,
                                            int16_t* values,
                                            bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<int32_t>::write_int32(int64_t num_items,
                                            int32_t* values,
                                            bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<int64_t>::write_int64(int64_t num_items,
                                            int64_t* values,
                                            bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<uint8_t>::write_uint8(int64_t num_items,
                                            uint8_t* values,
                                            bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<uint16_t>::write_uint16(int64_t num_items,
                                              uint16_t* values,
                                              bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<uint32_t>::write_uint32(int64_t num_items,
                                              uint32_t* values,
                                              bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<uint64_t>::write_uint64(int64_t num_items,
                                              uint64_t* values,
                                              bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<float>::write_float32(int64_t num_items,
                                            float* values,
                                            bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <>
  void
  ForthOutputBufferOf<double>::write_float64(int64_t num_items,
                                             double* values,
                                             bool byteswap) {  // noexcept
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::maybe_resize(int64_t next) {
    throw std::runtime_error(std::string("not implemented") + FILENAME(__LINE__));
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
