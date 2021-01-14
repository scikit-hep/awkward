// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/forth/ForthOutputBuffer.cpp", line)

#include <cmath>

#include "awkward/kernel-dispatch.h"

#include "awkward/array/NumpyArray.h"
#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  ////////// helper functions

  void byteswap16(int64_t num_items, void* ptr) {
    while (num_items != 0) {
      uint16_t value = *reinterpret_cast<uint16_t*>(ptr);
      *reinterpret_cast<uint16_t*>(ptr) = ((value >> 8) & 0x00ff) |
                                          ((value << 8) & 0xff00);
      ptr = reinterpret_cast<void*>(reinterpret_cast<size_t>(ptr) + 2);
      num_items--;
    }
  }

  void byteswap32(int64_t num_items, void* ptr) {
    while (num_items != 0) {
      uint32_t value = *reinterpret_cast<uint32_t*>(ptr);
      *reinterpret_cast<uint32_t*>(ptr) = ((value >> 24) & 0x000000ff) |
                                          ((value >>  8) & 0x0000ff00) |
                                          ((value <<  8) & 0x00ff0000) |
                                          ((value << 24) & 0xff000000);
      ptr = reinterpret_cast<void*>(reinterpret_cast<size_t>(ptr) + 4);
      num_items--;
    }
  }

  void byteswap64(int64_t num_items, void* ptr) {
    while (num_items != 0) {
      uint64_t value = *reinterpret_cast<uint64_t*>(ptr);
      *reinterpret_cast<uint64_t*>(ptr) = ((value >> 56) & 0x00000000000000ff) |
                                          ((value >> 40) & 0x000000000000ff00) |
                                          ((value >> 24) & 0x0000000000ff0000) |
                                          ((value >>  8) & 0x00000000ff000000) |
                                          ((value <<  8) & 0x000000ff00000000) |
                                          ((value << 24) & 0x0000ff0000000000) |
                                          ((value << 40) & 0x00ff000000000000) |
                                          ((value << 56) & 0xff00000000000000);
      ptr = reinterpret_cast<void*>(reinterpret_cast<size_t>(ptr) + 8);
      num_items--;
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
  const ContentPtr
  ForthOutputBufferOf<OUT>::toNumpyArray() const {
    util::dtype dtype;
    if (std::is_same<OUT, bool>::value) {
      dtype = util::dtype::boolean;
    }
    else if (std::is_same<OUT, int8_t>::value) {
      dtype = util::dtype::int8;
    }
    else if (std::is_same<OUT, int16_t>::value) {
      dtype = util::dtype::int16;
    }
    else if (std::is_same<OUT, int32_t>::value) {
      dtype = util::dtype::int32;
    }
    else if (std::is_same<OUT, int64_t>::value) {
      dtype = util::dtype::int64;
    }
    else if (std::is_same<OUT, uint8_t>::value) {
      dtype = util::dtype::uint8;
    }
    else if (std::is_same<OUT, uint16_t>::value) {
      dtype = util::dtype::uint16;
    }
    else if (std::is_same<OUT, uint32_t>::value) {
      dtype = util::dtype::uint32;
    }
    else if (std::is_same<OUT, uint64_t>::value) {
      dtype = util::dtype::uint64;
    }
    else if (std::is_same<OUT, float>::value) {
      dtype = util::dtype::float32;
    }
    else if (std::is_same<OUT, double>::value) {
      dtype = util::dtype::float64;
    }
    else {
      throw std::runtime_error(
        std::string("unrecognized ForthOutputBuffer specialization: ")
        + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
      );
    }
    return std::make_shared<NumpyArray>(Identities::none(),
                                        util::Parameters(),
                                        ptr_,
                                        std::vector<ssize_t>({ (ssize_t)length_ }),
                                        std::vector<ssize_t>({ (ssize_t)sizeof(OUT) }),
                                        0,
                                        (ssize_t)sizeof(OUT),
                                        util::dtype_to_format(dtype),
                                        dtype,
                                        kernel::lib::cpu);
  }

  template <typename OUT>
  const Index8
  ForthOutputBufferOf<OUT>::toIndex8() const {
    throw std::runtime_error(
      std::string("ForthOutputBuffer type is incompatible with Index8: ")
      + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
    );
  }

  template <typename OUT>
  const IndexU8
  ForthOutputBufferOf<OUT>::toIndexU8() const {
    throw std::runtime_error(
      std::string("ForthOutputBuffer type is incompatible with IndexU8: ")
      + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
    );
  }

  template <typename OUT>
  const Index32
  ForthOutputBufferOf<OUT>::toIndex32() const {
    throw std::runtime_error(
      std::string("ForthOutputBuffer type is incompatible with Index32: ")
      + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
    );
  }

  template <typename OUT>
  const IndexU32
  ForthOutputBufferOf<OUT>::toIndexU32() const {
    throw std::runtime_error(
      std::string("ForthOutputBuffer type is incompatible with IndexU32: ")
      + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
    );
  }

  template <typename OUT>
  const Index64
  ForthOutputBufferOf<OUT>::toIndex64() const {
    throw std::runtime_error(
      std::string("ForthOutputBuffer type is incompatible with Index64: ")
      + std::string(typeid(OUT).name()) + FILENAME(__LINE__)
    );
  }

  template <>
  const Index8
  ForthOutputBufferOf<int8_t>::toIndex8() const {
    return Index8(ptr_, 0, length_, kernel::lib::cpu);
  }

  template <>
  const IndexU8
  ForthOutputBufferOf<uint8_t>::toIndexU8() const {
    return IndexU8(ptr_, 0, length_, kernel::lib::cpu);
  }

  template <>
  const Index32
  ForthOutputBufferOf<int32_t>::toIndex32() const {
    return Index32(ptr_, 0, length_, kernel::lib::cpu);
  }

  template <>
  const IndexU32
  ForthOutputBufferOf<uint32_t>::toIndexU32() const {
    return IndexU32(ptr_, 0, length_, kernel::lib::cpu);
  }

  template <>
  const Index64
  ForthOutputBufferOf<int64_t>::toIndex64() const {
    return Index64(ptr_, 0, length_, kernel::lib::cpu);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_bool(bool value, bool byteswap) noexcept {
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int8(int8_t value, bool byteswap) noexcept {
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int16(int16_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int32(int32_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_int64(int64_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_intp(ssize_t value, bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(ssize_t) == 4) {
        byteswap32(1, &value);
      }
      else {
        byteswap64(1, &value);
      }
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint8(uint8_t value, bool byteswap) noexcept {
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint16(uint16_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint32(uint32_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uint64(uint64_t value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_uintp(size_t value, bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(size_t) == 4) {
        byteswap32(1, &value);
      }
      else {
        byteswap64(1, &value);
      }
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_float32(float value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_one_float64(double value, bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(1, &value);
    }
    write_one(value);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_bool(int64_t num_items,
                                       bool* values,
                                       bool byteswap) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int8(int64_t num_items,
                                       int8_t* values,
                                       bool byteswap) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int16(int64_t num_items,
                                        int16_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap16(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int32(int64_t num_items,
                                        int32_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap32(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_int64(int64_t num_items,
                                        int64_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap64(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_intp(int64_t num_items,
                                       ssize_t* values,
                                       bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(ssize_t) == 4) {
        byteswap32(num_items, values);
      }
      else {
        byteswap64(num_items, values);
      }
    }
    write_copy(num_items, values);
    if (byteswap) {
      if (sizeof(ssize_t) == 4) {
        byteswap32(num_items, values);
      }
      else {
        byteswap64(num_items, values);
      }
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint8(int64_t num_items,
                                        uint8_t* values,
                                        bool byteswap) noexcept {
    write_copy(num_items, values);
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint16(int64_t num_items,
                                         uint16_t* values,
                                         bool byteswap) noexcept {
    if (byteswap) {
      byteswap16(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap16(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint32(int64_t num_items,
                                         uint32_t* values,
                                         bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap32(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uint64(int64_t num_items,
                                         uint64_t* values,
                                         bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap64(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_uintp(int64_t num_items,
                                        size_t* values,
                                        bool byteswap) noexcept {
    if (byteswap) {
      if (sizeof(size_t) == 4) {
        byteswap32(num_items, values);
      }
      else {
        byteswap64(num_items, values);
      }
    }
    write_copy(num_items, values);
    if (byteswap) {
      if (sizeof(size_t) == 4) {
        byteswap32(num_items, values);
      }
      else {
        byteswap64(num_items, values);
      }
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float32(int64_t num_items,
                                          float* values,
                                          bool byteswap) noexcept {
    if (byteswap) {
      byteswap32(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap32(num_items, values);
    }
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::write_float64(int64_t num_items,
                                          double* values,
                                          bool byteswap) noexcept {
    if (byteswap) {
      byteswap64(num_items, values);
    }
    write_copy(num_items, values);
    if (byteswap) {
      byteswap64(num_items, values);
    }
  }

  template <>
  void
  ForthOutputBufferOf<bool>::write_bool(int64_t num_items,
                                        bool* values,
                                        bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(bool) * num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int8_t>::write_int8(int64_t num_items,
                                          int8_t* values,
                                          bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(int8_t) * num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<int16_t>::write_int16(int64_t num_items,
                                            int16_t* values,
                                            bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(int16_t) * num_items);
    if (byteswap) {
      byteswap16(num_items, &ptr_.get()[length_]);
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
    std::memcpy(&ptr_.get()[length_], values, sizeof(int32_t) * num_items);
    if (byteswap) {
      byteswap32(num_items, &ptr_.get()[length_]);
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
    std::memcpy(&ptr_.get()[length_], values, sizeof(int64_t) * num_items);
    if (byteswap) {
      byteswap64(num_items, &ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint8_t>::write_uint8(int64_t num_items,
                                            uint8_t* values,
                                            bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint8_t) * num_items);
    length_ = next;
  }

  template <>
  void
  ForthOutputBufferOf<uint16_t>::write_uint16(int64_t num_items,
                                              uint16_t* values,
                                              bool byteswap) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint16_t) * num_items);
    if (byteswap) {
      byteswap16(num_items, &ptr_.get()[length_]);
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
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint32_t) * num_items);
    if (byteswap) {
      byteswap32(num_items, &ptr_.get()[length_]);
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
    std::memcpy(&ptr_.get()[length_], values, sizeof(uint64_t) * num_items);
    if (byteswap) {
      byteswap64(num_items, &ptr_.get()[length_]);
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
    std::memcpy(&ptr_.get()[length_], values, sizeof(float) * num_items);
    if (byteswap) {
      byteswap32(num_items, &ptr_.get()[length_]);
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
    std::memcpy(&ptr_.get()[length_], values, sizeof(double) * num_items);
    if (byteswap) {
      byteswap64(num_items, &ptr_.get()[length_]);
    }
    length_ = next;
  }

  template <typename OUT>
  void
  ForthOutputBufferOf<OUT>::maybe_resize(int64_t next) {
    if (next > reserved_) {
      int64_t reservation = reserved_;
      while (next > reservation) {
        reservation = (int64_t)std::ceil(reservation * resize_);
      }
      std::shared_ptr<OUT> new_buffer = std::shared_ptr<OUT>(new OUT[reservation],
                                                             kernel::array_deleter<OUT>());
      std::memcpy(new_buffer.get(), ptr_.get(), sizeof(OUT) * reserved_);
      ptr_ = new_buffer;
      reserved_ = reservation;
    }
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
