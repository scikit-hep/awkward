// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthOutputBuffer.cpp", line)

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <type_traits>

#if defined(_MSC_VER)
#  include <stdlib.h>
#endif
#if __has_include(<bit>)
#  include <bit>
#endif

#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  ////////// helper functions

  namespace {
    // std::byteswap is C++23 *and* needs libstdc++ >= 12 / libc++ >= 14 /
    // MSVC 19.31. Older toolchains (e.g. GCC 11 on ubuntu-22.04, manylinux)
    // accept -std=c++23 but lack it, so fall back to the compiler builtins.
    inline uint16_t bswap_uint(uint16_t x) noexcept {
#if defined(__cpp_lib_byteswap)
      return std::byteswap(x);
#elif defined(_MSC_VER)
      return _byteswap_ushort(x);
#else
      return __builtin_bswap16(x);
#endif
    }

    inline uint32_t bswap_uint(uint32_t x) noexcept {
#if defined(__cpp_lib_byteswap)
      return std::byteswap(x);
#elif defined(_MSC_VER)
      return _byteswap_ulong(x);
#else
      return __builtin_bswap32(x);
#endif
    }

    inline uint64_t bswap_uint(uint64_t x) noexcept {
#if defined(__cpp_lib_byteswap)
      return std::byteswap(x);
#elif defined(_MSC_VER)
      return _byteswap_uint64(x);
#else
      return __builtin_bswap64(x);
#endif
    }

    // Byte-swap num_items consecutive values of width sizeof(UINT) in place,
    // starting at &value. Forth input buffers may be unaligned, so values are
    // copied through memcpy rather than dereferenced.
    template <typename UINT, typename T>
    void byteswap_n(int64_t num_items, T& value) noexcept {
      static_assert(sizeof(T) == sizeof(UINT),
                    "byteswap width does not match the element type");
      static_assert(std::is_trivially_copyable<T>::value,
                    "byteswap requires a trivially copyable type");
      auto* ptr = reinterpret_cast<unsigned char*>(&value);
      for (int64_t i = 0;  i < num_items;  i++) {
        UINT integer;
        std::memcpy(&integer, ptr, sizeof(UINT));
        integer = bswap_uint(integer);
        std::memcpy(ptr, &integer, sizeof(UINT));
        ptr += sizeof(UINT);
      }
    }
  }

  template <typename T>
  void byteswap16(int64_t num_items, T& value) {
    byteswap_n<uint16_t>(num_items, value);
  }

  template <typename T>
  void byteswap32(int64_t num_items, T& value) {
    byteswap_n<uint32_t>(num_items, value);
  }

  template <typename T>
  void byteswap64(int64_t num_items, T& value) {
    byteswap_n<uint64_t>(num_items, value);
  }

  template <typename T>
  void byteswap_intp(int64_t num_items, T& value) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "byteswap_intp requires a 32- or 64-bit integer");
    if constexpr (sizeof(T) == 4) {
      byteswap32(num_items, value);
    }
    else {
      byteswap64(num_items, value);
    }
  }

  ////////// abstract

  ForthOutputBuffer::ForthOutputBuffer(int64_t initial, double resize)
    : length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  ForthOutputBuffer::~ForthOutputBuffer() = default;

  int64_t
  ForthOutputBuffer::len() const noexcept {
    return length_;
  }

  void
  ForthOutputBuffer::rewind(int64_t num_items, util::ForthError& err) noexcept {
    int64_t next = length_ - num_items;
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
  ForthOutputBufferOf<OUT>::ForthOutputBufferOf(int64_t initial, double resize)
    : ForthOutputBuffer(initial, resize)
    , ptr_(new OUT[(size_t)initial], util::array_deleter<OUT>()) { }

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
  ForthOutputBufferOf<OUT>::dup(int64_t num_times, util::ForthError& err) noexcept {
    if (length_ == 0) {
      err = util::ForthError::rewind_beyond;
    }
    else if (num_times > 0) {
      int64_t next = length_ + num_times;
      maybe_resize(next);
      OUT value = ptr_.get()[length_ - 1];
      for (int64_t i = 0;  i < num_times;  i++) {
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
      byteswap_intp(1, value);
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
      byteswap_intp(1, value);
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
  ForthOutputBufferOf<OUT>::write_one_string(char* string_buffer, int64_t length) noexcept {
    int64_t next = length_ + length;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], string_buffer, (size_t)length);
    length_ = next;
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_bool(int64_t num_items,
                                       bool* values,
                                       bool /* byteswap */) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int8(int64_t num_items,
                                       int8_t* values,
                                       bool /* byteswap */) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int16(int64_t num_items,
                                        int16_t* values,
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
  ForthOutputBufferOf<OUT>::write_int32(int64_t num_items,
                                        int32_t* values,
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
  ForthOutputBufferOf<OUT>::write_int64(int64_t num_items,
                                        int64_t* values,
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
  ForthOutputBufferOf<OUT>::write_intp(int64_t num_items,
                                       ssize_t* values,
                                       bool byteswap) noexcept {
    if (byteswap) {
      byteswap_intp(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap_intp(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_const_uint8(int64_t num_items,
                                              const uint8_t* values) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint8(int64_t num_items,
                                        uint8_t* values,
                                        bool /* byteswap */) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint16(int64_t num_items,
                                         uint16_t* values,
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
  ForthOutputBufferOf<OUT>::write_uint32(int64_t num_items,
                                         uint32_t* values,
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
  ForthOutputBufferOf<OUT>::write_uint64(int64_t num_items,
                                         uint64_t* values,
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
  ForthOutputBufferOf<OUT>::write_uintp(int64_t num_items,
                                        size_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap_intp(num_items, *values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap_intp(num_items, *values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float32(int64_t num_items,
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
  ForthOutputBufferOf<OUT>::write_float64(int64_t num_items,
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
  ForthOutputBufferOf<bool>::write_bool(int64_t num_items,
                                        bool* values,
                                        bool /* byteswap */) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(bool) * (size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int8_t>::write_int8(int64_t num_items,
                                          int8_t* values,
                                          bool /* byteswap */) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(int8_t) * (size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int16_t>::write_int16(int64_t num_items,
                                            int16_t* values,
                                            bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(int16_t) * (size_t)num_items);
    if (byteswap) {
      byteswap16(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int32_t>::write_int32(int64_t num_items,
                                            int32_t* values,
                                            bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(int32_t) * (size_t)num_items);
    if (byteswap) {
      byteswap32(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int64_t>::write_int64(int64_t num_items,
                                            int64_t* values,
                                            bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(int64_t) * (size_t)num_items);
    if (byteswap) {
      byteswap64(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint8_t>::write_const_uint8(int64_t num_items,
                                                  const uint8_t* values) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint8_t) * (size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint8_t>::write_uint8(int64_t num_items,
                                            uint8_t* values,
                                            bool /* byteswap */) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint8_t) * (size_t)num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint16_t>::write_uint16(int64_t num_items,
                                              uint16_t* values,
                                              bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint16_t) * (size_t)num_items);
    if (byteswap) {
      byteswap16(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint32_t>::write_uint32(int64_t num_items,
                                              uint32_t* values,
                                              bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint32_t) * (size_t)num_items);
    if (byteswap) {
      byteswap32(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint64_t>::write_uint64(int64_t num_items,
                                              uint64_t* values,
                                              bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint64_t) * (size_t)num_items);
    if (byteswap) {
      byteswap64(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<float>::write_float32(int64_t num_items,
                                            float* values,
                                            bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(float) * (size_t)num_items);
    if (byteswap) {
      byteswap32(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<double>::write_float64(int64_t num_items,
                                             double* values,
                                             bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(double) * (size_t)num_items);
    if (byteswap) {
      byteswap64(num_items, ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_add_int32(int32_t value) noexcept {
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
  ForthOutputBufferOf<OUT>::write_add_int64(int64_t value) noexcept {
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
  ForthOutputBufferOf<OUT>::maybe_resize(int64_t next) {
    if (next > reserved_) {
      int64_t reservation = reserved_;
      while (next > reservation) {
        int64_t grown = (int64_t)std::ceil((double)reservation * (double)resize_);
        // Guarantee strict growth: when reservation is 0 or resize_ <= 1.0 the
        // geometric step can fail to increase reservation, looping forever.
        if (grown <= reservation) {
          grown = reservation + 1;
        }
        reservation = grown;
      }
      std::shared_ptr<OUT> new_buffer = std::shared_ptr<OUT>(new OUT[(size_t)reservation],
                                                             util::array_deleter<OUT>());
      std::memcpy(new_buffer.get(), ptr_.get(), sizeof(OUT) * (size_t)reserved_);
      ptr_ = new_buffer;
      reserved_ = reservation;
    }
  }

  template void byteswap16<int16_t>(int64_t num_items, int16_t& value);
  template void byteswap16<uint16_t>(int64_t num_items, uint16_t& value);
  template void byteswap32<int32_t>(int64_t num_items, int32_t& value);
  template void byteswap32<uint32_t>(int64_t num_items, uint32_t& value);
  template void byteswap32<float>(int64_t num_items, float& value);
  template void byteswap64<int64_t>(int64_t num_items, int64_t& value);
  template void byteswap64<uint64_t>(int64_t num_items, uint64_t& value);
  template void byteswap64<double>(int64_t num_items, double& value);

  template void byteswap_intp<size_t>(int64_t num_items, size_t& value);
  template void byteswap_intp<ssize_t>(int64_t num_items, ssize_t& value);

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
