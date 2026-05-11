// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthOutputBuffer.cpp", line)

#include <cmath>
#include <cstring>
#include <sstream>

#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  ////////// helper functions

  template <typename T>
  void byteswap16(std::int64_t num_items, T& value) {
    T* ptr = &value;
    while (num_items != 0) {
      uint16_t integer;
      std::memcpy(&integer, ptr, sizeof(std::uint16_t));
      uint16_t output = ((integer >> 8) & 0x00ff) |
                        ((integer << 8) & 0xff00);
      std::memcpy(ptr, &output, sizeof(T));
      ptr = reinterpret_cast<T*>(reinterpret_cast<std::size_t>(ptr) + 2);
      num_items--;
    }
  }

  template <typename T>
  void byteswap32(std::int64_t num_items, T& value) {
    T* ptr = &value;
    while (num_items != 0) {
      uint32_t integer;
      std::memcpy(&integer, ptr, sizeof(std::uint32_t));
      uint32_t output = ((integer >> 24) & 0x000000ff) |
                        ((integer >>  8) & 0x0000ff00) |
                        ((integer <<  8) & 0x00ff0000) |
                        ((integer << 24) & 0xff000000);
      std::memcpy(ptr, &output, sizeof(T));
      ptr = reinterpret_cast<T*>(reinterpret_cast<std::size_t>(ptr) + 4);
      num_items--;
    }
  }

  template <typename T>
  void byteswap64(std::int64_t num_items, T& value) {
    T* ptr = &value;
    while (num_items != 0) {
      std::uint64_t integer;
      std::memcpy(&integer, ptr, sizeof(std::uint64_t));
      std::uint64_t output = ((integer >> 56) & 0x00000000000000ff) |
                        ((integer >> 40) & 0x000000000000ff00) |
                        ((integer >> 24) & 0x0000000000ff0000) |
                        ((integer >>  8) & 0x00000000ff000000) |
                        ((integer <<  8) & 0x000000ff00000000) |
                        ((integer << 24) & 0x0000ff0000000000) |
                        ((integer << 40) & 0x00ff000000000000) |
                        ((integer << 56) & 0xff00000000000000);
      std::memcpy(ptr, &output, sizeof(T));
      ptr = reinterpret_cast<T*>(reinterpret_cast<std::size_t>(ptr) + 8);
      num_items--;
    }
  }

  template <typename T>
  void byteswap_intp(std::int64_t num_items, T& value) {
    T* ptr = &value;
    if (sizeof(ssize_t) == 4) {
      while (num_items != 0) {
        std::uint32_t integer;
        std::memcpy(&integer, ptr, sizeof(std::uint32_t));
        std::uint32_t output = ((integer >> 24) & 0x000000ff) |
                          ((integer >>  8) & 0x0000ff00) |
                          ((integer <<  8) & 0x00ff0000) |
                          ((integer << 24) & 0xff000000);
        std::memcpy(ptr, &output, sizeof(T));
        ptr = reinterpret_cast<T*>(reinterpret_cast<std::size_t>(ptr) + 4);
        num_items--;
      }
    }
    else {
      while (num_items != 0) {
        std::uint64_t integer = *reinterpret_cast<std::uint64_t*>(ptr);
        std::uint64_t output = ((integer >> 56) & 0x00000000000000ff) |
                          ((integer >> 40) & 0x000000000000ff00) |
                          ((integer >> 24) & 0x0000000000ff0000) |
                          ((integer >>  8) & 0x00000000ff000000) |
                          ((integer <<  8) & 0x000000ff00000000) |
                          ((integer << 24) & 0x0000ff0000000000) |
                          ((integer << 40) & 0x00ff000000000000) |
                          ((integer << 56) & 0xff00000000000000);
        std::memcpy(ptr, &output, sizeof(T));
        ptr = reinterpret_cast<T*>(reinterpret_cast<size_t>(ptr) + 8);
        num_items--;
      }
    }
  }

  ////////// abstract

  ForthOutputBuffer::ForthOutputBuffer(std::int64_t initial, double resize)
    : length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  ForthOutputBuffer::~ForthOutputBuffer() = default;

  std::int64_t
  ForthOutputBuffer::len() const noexcept {
    return length_;
  }

  void
  ForthOutputBuffer::rewind(std::int64_t num_items, util::ForthError& err) noexcept {
    std::int64_t next = length_ - num_items;
    if (next < 0) {
      err = util::ForthError::rewind_beyond;
    }
    else {
      length_ = next;
    }
  }

  void
  ForthOutputBuffer::reset() noexcept {
    length_ = 0;
  }

  ////////// specialized

  template <typename OUT>
  ForthOutputBufferOf<OUT>::ForthOutputBufferOf(std::int64_t initial, double resize)
    : ForthOutputBuffer(initial, resize)
    , ptr_(new OUT[(std::size_t)initial], util::array_deleter<OUT>()) { }

  template <typename OUT>
  const std::shared_ptr<void>
  ForthOutputBufferOf<OUT>::ptr() const noexcept {
    return ptr_;
  }

  template <typename OUT>
  util::dtype
  ForthOutputBufferOf<OUT>::dtype() const {
    if (std::is_same<OUT, bool>::value) {
      return util::dtype::boolean;
    }
    else if (std::is_same<OUT, std::int8_t>::value) {
      return util::dtype::int8;
    }
    else if (std::is_same<OUT, std::int16_t>::value) {
      return util::dtype::int16;
    }
    else if (std::is_same<OUT, std::int32_t>::value) {
      return util::dtype::int32;
    }
    else if (std::is_same<OUT, std::int64_t>::value) {
      return util::dtype::int64;
    }
    else if (std::is_same<OUT, std::uint8_t>::value) {
      return util::dtype::uint8;
    }
    else if (std::is_same<OUT, std::uint16_t>::value) {
      return util::dtype::uint16;
    }
    else if (std::is_same<OUT, std::uint32_t>::value) {
      return util::dtype::uint32;
    }
    else if (std::is_same<OUT, std::uint64_t>::value) {
      return util::dtype::uint64;
    }
    else if (std::is_same<OUT, float>::value) {
      return util::dtype::float32;
    }
    else if (std::is_same<OUT, double>::value) {
      return util::dtype::float64;
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized ForthOutputBuffer specialization: ")
        + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
      );
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::dup(std::int64_t num_times, util::ForthError& err) noexcept {
    if (length_ == 0) {
      err = util::ForthError::rewind_beyond;
    }
    else if (num_times > 0) {
      std::int64_t next = length_ + num_times;
      maybe_resize(next);
      OUT value = ptr_.get()[length_ - 1];
      for (std::int64_t i = 0;  i < num_times;  i++) {
        ptr_.get()[length_ + i] = value;
      }
      length_ = next;
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_bool(bool value, bool /* byteswap */) noexcept {
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int8(int8_t value, bool /* byteswap */) noexcept {
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int16(int16_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int32(int32_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int64(int64_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_intp(ssize_t value, bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(ssize_t) == 4) {
        byteswap32(1, value);
      }
      else {
        byteswap64(1, value);
      }
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint8(uint8_t value, bool /* byteswap */) noexcept {
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint16(uint16_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint32(uint32_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint64(uint64_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uintp(size_t value, bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(size_t) == 4) {
        byteswap32(1, value);
      }
      else {
        byteswap64(1, value);
      }
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_float32(float value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(1, value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_float64(double value, bool byteswap) noexcept {
     if (byteswap) {
      byteswap64(1, value);
    }
    write_one(value);
}

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_string(char* string_buffer, std::int64_t length) noexcept {
    std::int64_t next = length_ + length;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], string_buffer, (std::size_t)length);
    length_ = next;
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_bool(std::int64_t num_items,
                                       bool* values,
                                       bool /* byteswap */) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int8(std::int64_t num_items,
                                       std::int8_t* values,
                                       bool /* byteswap */) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int16(std::int64_t num_items,
                                        std::int16_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap16(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int32(std::int64_t num_items,
                                        std::int32_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap32(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int64(std::int64_t num_items,
                                        std::int64_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap64(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_intp(std::int64_t num_items,
                                       ssize_t* values,
                                       bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(ssize_t) == 4) {
        byteswap32(num_items, *values);
      }
      else {
        byteswap64(num_items, *values);
      }
    }
    write_copy(num_items, values);
    if (byteswap) {
      if (sizeof(ssize_t) == 4) {
        byteswap32(num_items, *values);
      }
      else {
        byteswap64(num_items, *values);
      }
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_const_uint8(std::int64_t num_items,
                                              const std::uint8_t* values) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint8(std::int64_t num_items,
                                        std::uint8_t* values,
                                        bool /* byteswap */) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint16(std::int64_t num_items,
                                         std::uint16_t* values,
                                         bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap16(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint32(std::int64_t num_items,
                                         std::uint32_t* values,
                                         bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap32(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint64(std::int64_t num_items,
                                         std::uint64_t* values,
                                         bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap64(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uintp(std::int64_t num_items,
                                        std::size_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(std::size_t) == 4) {
        byteswap32(num_items, *values);
      }
      else {
        byteswap64(num_items, *values);
      }
    }
    write_copy(num_items, values);
    if (byteswap) {
      if (sizeof(std::size_t) == 4) {
        byteswap32(num_items, *values);
      }
      else {
        byteswap64(num_items, *values);
      }
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float32(std::int64_t num_items,
                                          float* values,
                                          bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap32(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float64(std::int64_t num_items,
                                          double* values,
                                          bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap64(num_items, *values);
    }
  }

  template <>
  void
  ForthOutputBufferOf<bool>::write_bool(std::int64_t num_items,
                                        bool* values,
                                        bool /* byteswap */) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(bool) * (size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int8_t>::write_int8(std::int64_t num_items,
                                          std::int8_t* values,
                                          bool /* byteswap */) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::int8_t) * (std::size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::int16_t>::write_int16(std::int64_t num_items,
                                            std::int16_t* values,
                                            bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::int16_t) * (std::size_t)num_items);
    if (byteswap) {
      byteswap16(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::int32_t>::write_int32(std::int64_t num_items,
                                            std::int32_t* values,
                                            bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::int32_t) * (std::size_t)num_items);
    if (byteswap) {
      byteswap32(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::int64_t>::write_int64(std::int64_t num_items,
                                            std::int64_t* values,
                                            bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::int64_t) * (std::size_t)num_items);
    if (byteswap) {
      byteswap64(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::uint8_t>::write_const_uint8(std::int64_t num_items,
                                                  const std::uint8_t* values) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::uint8_t) * (std::size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::uint8_t>::write_uint8(std::int64_t num_items,
                                                 std::uint8_t* values,
                                                 bool /* byteswap */) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::uint8_t) * (std::size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::uint16_t>::write_uint16(std::int64_t num_items,
                                                    std::uint16_t* values,
                                                    bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::uint16_t) * (std::size_t)num_items);
    if (byteswap) {
      byteswap16(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::uint32_t>::write_uint32(std::int64_t num_items,
                                                    std::uint32_t* values,
                                              bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::uint32_t) * (std::size_t)num_items);
    if (byteswap) {
      byteswap32(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<std::uint64_t>::write_uint64(std::int64_t num_items,
                                                    std::uint64_t* values,
                                              bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(std::uint64_t) * (std::size_t)num_items);
    if (byteswap) {
      byteswap64(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<float>::write_float32(std::int64_t num_items,
                                            float* values,
                                            bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(float) * (std::size_t)num_items);
    if (byteswap) {
      byteswap32(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<double>::write_float64(std::int64_t num_items,
                                             double* values,
                                             bool byteswap) noexcept {
    std::int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(double) * (std::size_t)num_items);
    if (byteswap) {
      byteswap64(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_add_int32(std::int32_t value) noexcept {
    OUT previous = 0;
    if (length_ != 0) {
      previous = ptr_.get()[length_ - 1];
    }
    length_++;
    maybe_resize(length_);
    ptr_.get()[length_ - 1] = previous + (OUT)value;
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_add_int64(std::int64_t value) noexcept {
    OUT previous = 0;
    if (length_ != 0) {
      previous = ptr_.get()[length_ - 1];
    }
    length_++;
    maybe_resize(length_);
    ptr_.get()[length_ - 1] = previous + (OUT)value;
  }

  template <typename OUT>
  std::string
  ForthOutputBufferOf<OUT>::tostring() const {
    std::stringstream ss;
    ss << "[";
    if (length_ > 0) {
      ss << ptr_.get()[0];
      for (auto i = 1; i < length_; i++) {
        ss << ", ";
        ss << ptr_.get()[i];
      }
    }
    ss << "]";
    return ss.str();
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::maybe_resize(std::int64_t next) {
    if (next > reserved_) {
      std::int64_t reservation = reserved_;
      while (next > reservation) {
        reservation = (std::int64_t)std::ceil((double)reservation * (double)resize_);
      }
      std::shared_ptr<OUT> new_buffer = std::shared_ptr<OUT>(new OUT[(std::size_t)reservation],
                                                             util::array_deleter<OUT>());
      std::memcpy(new_buffer.get(), ptr_.get(), sizeof(OUT) * (std::size_t)reserved_);
      ptr_ = new_buffer;
      reserved_ = reservation;
    }
  }

  template void byteswap16<std::int16_t>(std::int64_t num_items, std::int16_t& value);
  template void byteswap16<std::uint16_t>(std::int64_t num_items, std::uint16_t& value);
  template void byteswap32<std::int32_t>(std::int64_t num_items, std::int32_t& value);
  template void byteswap32<std::uint32_t>(std::int64_t num_items, std::uint32_t& value);
  template void byteswap32<float>(std::int64_t num_items, float& value);
  template void byteswap64<std::int64_t>(std::int64_t num_items, std::int64_t& value);
  template void byteswap64<std::uint64_t>(std::int64_t num_items, std::uint64_t& value);
  template void byteswap64<double>(std::int64_t num_items, double& value);

  template void byteswap_intp<std::size_t>(std::int64_t num_items, std::size_t& value);
  template void byteswap_intp<ssize_t>(std::int64_t num_items, ssize_t& value);

  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<bool>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::int8_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::int16_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::int32_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::int64_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::uint8_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::uint16_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::uint32_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<std::uint64_t>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<float>;
  template class EXPORT_TEMPLATE_INST ForthOutputBufferOf<double>;

}
