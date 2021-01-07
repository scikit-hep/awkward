// c++ virtual-machine.cpp -O5 -o virtual-machine-test  &&  echo GO  &&  ./virtual-machine-test  &&  ./virtual-machine-test  &&  ./virtual-machine-test  &&  ./virtual-machine-test  &&  ./virtual-machine-test

#include <memory>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <cstring>

#include <chrono>
#include <iostream>


template <typename T>
class array_deleter {
public:
    void operator()(T const *ptr) {
      delete [] ptr;
    }
};


enum class dtype {
    NOT_PRIMITIVE,
    boolean,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    float128,
    complex64,
    complex128,
    complex256,
    // datetime64,
    // timedelta64,
    size
};


enum class ForthError {
  none,
  stack_underflow,
  read_beyond,
  seek_beyond,
  skip_beyond,
  rewind_beyond,
  size
};


#define NATIVELY_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

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


class ForthInputBuffer {
public:
  ForthInputBuffer(const std::shared_ptr<void> ptr,
                   int64_t offset,
                   int64_t length)
    : ptr_(ptr)
    , offset_(offset)
    , length_(length)
    , pos_(0) { }

  void* read(int64_t num_bytes, ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next > length_) {
      err = ForthError::read_beyond;
      return nullptr;
    }
    void* out = reinterpret_cast<void*>(
        reinterpret_cast<size_t>(ptr_.get()) + (size_t)offset_ + (size_t)pos_
    );
    pos_ = next;
    return out;
  }

  void seek(int64_t to, ForthError& err) noexcept {
    if (to < 0  ||  to > length_) {
      err = ForthError::seek_beyond;
    }
    else {
      pos_ = to;
    }
  }

  void skip(int64_t num_bytes, ForthError& err) noexcept {
    int64_t next = pos_ + num_bytes;
    if (next < 0  ||  next > length_) {
      err = ForthError::skip_beyond;
    }
    else {
      pos_ = next;
    }
  }

  bool end() const noexcept {
    return pos_ == length_;
  }

  int64_t pos() const noexcept {
    return pos_;
  }

  int64_t len() const noexcept {
    return length_;
  }

private:
  std::shared_ptr<void> ptr_;
  int64_t offset_;
  int64_t length_;
  int64_t pos_;
};


class ForthOutputBuffer {
public:
  ForthOutputBuffer(int64_t initial=1024, double resize=1.5)
    : length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  int64_t length() const {
    return length_;
  }

  void rewind(int64_t num_items, ForthError& err) noexcept {
    int64_t next = length_ - num_items;
    if (next < 0) {
      err = ForthError::rewind_beyond;
    }
    else {
      length_ = next;
    }
  }

  virtual std::shared_ptr<void> ptr() const noexcept = 0;

  virtual inline void write_one_bool(bool value, bool byteswap) noexcept = 0;
  virtual inline void write_one_int8(int8_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_int16(int16_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_int32(int32_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_int64(int64_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_intp(ssize_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_uint8(uint8_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_uint16(uint16_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_uint32(uint32_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_uint64(uint64_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_uintp(size_t value, bool byteswap) noexcept = 0;
  virtual inline void write_one_float32(float value, bool byteswap) noexcept = 0;
  virtual inline void write_one_float64(double value, bool byteswap) noexcept = 0;

  virtual inline void write_bool(int64_t num_items, bool* values, bool byteswap) noexcept = 0;
  virtual inline void write_int8(int64_t num_items, int8_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_int16(int64_t num_items, int16_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_int32(int64_t num_items, int32_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_int64(int64_t num_items, int64_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_intp(int64_t num_items, ssize_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_uint8(int64_t num_items, uint8_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_uint16(int64_t num_items, uint16_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_uint32(int64_t num_items, uint32_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_uint64(int64_t num_items, uint64_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_uintp(int64_t num_items, size_t* values, bool byteswap) noexcept = 0;
  virtual inline void write_float32(int64_t num_items, float* values, bool byteswap) noexcept = 0;
  virtual inline void write_float64(int64_t num_items, double* values, bool byteswap) noexcept = 0;

  virtual const std::string tostring() const = 0;

protected:
  int64_t length_;
  int64_t reserved_;
  double resize_;
};


template <typename OUT>
class ForthOutputBufferOf : public ForthOutputBuffer {
public:
  ForthOutputBufferOf(int64_t initial=1024, double resize=1.5)
    : ForthOutputBuffer(initial, resize)
    , ptr_(new OUT[initial], array_deleter<OUT>()) { }

  std::shared_ptr<void> ptr() const noexcept override {
    return ptr_;
  }

  inline void write_one_bool(bool value, bool byteswap) noexcept override {
    write_one(value);
  }
  inline void write_one_int8(int8_t value, bool byteswap) noexcept override {
    write_one(value);
  }
  inline void write_one_int16(int16_t value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap16(1, &value);
    }
    write_one(value);
  }
  inline void write_one_int32(int32_t value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap32(1, &value);
    }
    write_one(value);
  }
  inline void write_one_int64(int64_t value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap64(1, &value);
    }
    write_one(value);
  }
  inline void write_one_intp(ssize_t value, bool byteswap) noexcept override {
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
  inline void write_one_uint8(uint8_t value, bool byteswap) noexcept override {
    write_one(value);
  }
  inline void write_one_uint16(uint16_t value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap16(1, &value);
    }
    write_one(value);
  }
  inline void write_one_uint32(uint32_t value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap32(1, &value);
    }
    write_one(value);
  }
  inline void write_one_uint64(uint64_t value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap64(1, &value);
    }
    write_one(value);
  }
  inline void write_one_uintp(size_t value, bool byteswap) noexcept override {
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
  inline void write_one_float32(float value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap32(1, &value);
    }
    write_one(value);
  }
  inline void write_one_float64(double value, bool byteswap) noexcept override {
    if (byteswap) {
      byteswap64(1, &value);
    }
    write_one(value);
  }

  inline void write_bool(int64_t num_items, bool* values, bool byteswap) noexcept override;
  inline void write_int8(int64_t num_items, int8_t* values, bool byteswap) noexcept override;
  inline void write_int16(int64_t num_items, int16_t* values, bool byteswap) noexcept override;
  inline void write_int32(int64_t num_items, int32_t* values, bool byteswap) noexcept override;
  inline void write_int64(int64_t num_items, int64_t* values, bool byteswap) noexcept override;
  inline void write_intp(int64_t num_items, ssize_t* values, bool byteswap) noexcept override;
  inline void write_uint8(int64_t num_items, uint8_t* values, bool byteswap) noexcept override;
  inline void write_uint16(int64_t num_items, uint16_t* values, bool byteswap) noexcept override;
  inline void write_uint32(int64_t num_items, uint32_t* values, bool byteswap) noexcept override;
  inline void write_uint64(int64_t num_items, uint64_t* values, bool byteswap) noexcept override;
  inline void write_uintp(int64_t num_items, size_t* values, bool byteswap) noexcept override;
  inline void write_float32(int64_t num_items, float* values, bool byteswap) noexcept override;
  inline void write_float64(int64_t num_items, double* values, bool byteswap) noexcept override;

  const std::string tostring() const override {
    std::stringstream out;
    if (length_ <= 20) {
      for (int64_t i = 0;  i < length_;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << ptr_.get()[i];
      }
    }
    else {
      for (int64_t i = 0;  i < 10;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << ptr_.get()[i];
      }
      out << " ...";
      for (int64_t i = length_ - 10;  i < length_;  i++) {
        if (i != 0) {
          out << " ";
        }
        out << ptr_.get()[i];
      }
    }
    return out.str();
  }

private:
  void maybe_resize(int64_t next) {
    if (next > reserved_) {
      int64_t reservation = reserved_;
      while (next > reservation) {
        reservation = (int64_t)std::ceil(reservation * resize_);
      }
      std::shared_ptr<OUT> new_buffer = std::shared_ptr<OUT>(new OUT[reservation],
                                                             array_deleter<OUT>());
      std::memcpy(new_buffer.get(), ptr_.get(), sizeof(OUT) * reserved_);
      ptr_ = new_buffer;
      reserved_ = reservation;
    }
  }

  template <typename IN>
  inline void write_one(IN value) noexcept {
    length_++;
    maybe_resize(length_);
    ptr_.get()[length_ - 1] = value;
  }

  template <typename IN>
  inline void write_copy(int64_t num_items, const IN* values) noexcept {
    int64_t next = length_ + num_items;
    maybe_resize(next);
    for (int64_t i = 0;  i < num_items;  i++) {
      ptr_.get()[length_ + i] = values[i];
    }
    length_ = next;
  }

  std::shared_ptr<OUT> ptr_;
};


template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_bool(int64_t num_items, bool* values, bool byteswap) noexcept {
  write_copy(num_items, values);
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_int8(int64_t num_items, int8_t* values, bool byteswap) noexcept {
  write_copy(num_items, values);
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_int16(int64_t num_items, int16_t* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap16(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap16(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_int32(int64_t num_items, int32_t* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap32(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap32(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_int64(int64_t num_items, int64_t* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap64(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap64(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_intp(int64_t num_items, ssize_t* values, bool byteswap) noexcept {
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
inline void
ForthOutputBufferOf<OUT>::write_uint8(int64_t num_items, uint8_t* values, bool byteswap) noexcept {
  write_copy(num_items, values);
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_uint16(int64_t num_items, uint16_t* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap16(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap16(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_uint32(int64_t num_items, uint32_t* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap32(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap32(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_uint64(int64_t num_items, uint64_t* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap64(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap64(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_uintp(int64_t num_items, size_t* values, bool byteswap) noexcept {
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
inline void
ForthOutputBufferOf<OUT>::write_float32(int64_t num_items, float* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap32(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap32(num_items, values);
  }
}

template <typename OUT>
inline void
ForthOutputBufferOf<OUT>::write_float64(int64_t num_items, double* values, bool byteswap) noexcept {
  if (byteswap) {
    byteswap64(num_items, values);
  }
  write_copy(num_items, values);
  if (byteswap) {
    byteswap64(num_items, values);
  }
}

template <>
inline void
ForthOutputBufferOf<bool>::write_bool(int64_t num_items, bool* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(bool) * num_items);
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<int8_t>::write_int8(int64_t num_items, int8_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(int8_t) * num_items);
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<int16_t>::write_int16(int64_t num_items, int16_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(int16_t) * num_items);
  if (byteswap) {
    byteswap16(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<int32_t>::write_int32(int64_t num_items, int32_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(int32_t) * num_items);
  if (byteswap) {
    byteswap32(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<int64_t>::write_int64(int64_t num_items, int64_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(int64_t) * num_items);
  if (byteswap) {
    byteswap64(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<uint8_t>::write_uint8(int64_t num_items, uint8_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(uint8_t) * num_items);
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<uint16_t>::write_uint16(int64_t num_items, uint16_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(uint16_t) * num_items);
  if (byteswap) {
    byteswap16(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<uint32_t>::write_uint32(int64_t num_items, uint32_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(uint32_t) * num_items);
  if (byteswap) {
    byteswap32(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<uint64_t>::write_uint64(int64_t num_items, uint64_t* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(uint64_t) * num_items);
  if (byteswap) {
    byteswap64(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<float>::write_float32(int64_t num_items, float* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(float) * num_items);
  if (byteswap) {
    byteswap32(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}

template <>
inline void
ForthOutputBufferOf<double>::write_float64(int64_t num_items, double* values, bool byteswap) noexcept {
  int64_t next = length_ + num_items;
  maybe_resize(next);
  std::memcpy(&ptr_.get()[length_], values, sizeof(double) * num_items);
  if (byteswap) {
    byteswap64(num_items, &ptr_.get()[length_]);
  }
  length_ = next;
}


template <typename T>
class ForthStack {
public:
  ForthStack(int64_t initial=1024, double resize=1.5)
    : buffer_(new T[initial])
    , length_(0)
    , reserved_(initial)
    , resize_(resize) { }

  ~ForthStack() {
    delete [] buffer_;
  }

  int64_t length() const noexcept {
    return length_;
  }

  inline void push(T value) noexcept {
    if (length_ == reserved_) {
      int64_t reservation = (int64_t)std::ceil(reserved_ * resize_);
      T* new_buffer = new T[reservation];
      std::memcpy(new_buffer, buffer_, sizeof(T) * reserved_);
      delete [] buffer_;
      buffer_ = new_buffer;
      reserved_ = reservation;
    }
    buffer_[length_] = value;
    length_++;
  }

  inline void drop(ForthError &err) noexcept {
    if (length_ == 0) {
      err = ForthError::stack_underflow;
    }
    else {
      length_--;
    }
  }

  inline T pop(ForthError &err) noexcept {
    if (length_ == 0) {
      err = ForthError::stack_underflow;
      return 0;
    }
    else {
      length_--;
      return buffer_[length_];
    }
  }

  inline T* pop2(ForthError &err) noexcept {
    if (length_ < 2) {
      err = ForthError::stack_underflow;
      return 0;
    }
    else {
      length_ -= 2;
      return &buffer_[length_];
    }
  }

  inline T* peek() const noexcept {
    if (length_ == 0) {
      return nullptr;
    }
    else {
      return &buffer_[length_ - 1];
    }
  }

  void clear() noexcept {
    length_ = 0;
  }

  const std::string tostring() const {
    std::stringstream out;
    int64_t i = length_ - 20;
    if (i <= 0) {
      i = 0;
    }
    else {
      out << "... ";
    }
    for (;  i < length_;  i++) {
      out << buffer_[i] << " ";
    }
    if (length_ == 0) {
      out << "(empty)";
    }
    else {
      out << "<- top";
    }
    return out.str();
  }

  const std::vector<T> values() const {
    std::vector<T> out;
    for (int64_t i = 0;  i < length_;  i++) {
      out.push_back(buffer_[i]);
    }
    return out;
  }

private:
  T* buffer_;
  int64_t length_;
  int64_t reserved_;
  double resize_;
};


class ForthInstructionPointer {
public:
  ForthInstructionPointer(int64_t reservation=1024)
    : which_(new int64_t[reservation])
    , where_(new int64_t[reservation])
    , skip_(new int64_t[reservation])
    , length_(0)
    , reserved_(reservation) { }

  ~ForthInstructionPointer() {
    delete [] which_;
    delete [] where_;
    delete [] skip_;
  }

  inline bool empty() const noexcept {
    return length_ == 0;
  }

  inline bool push(int64_t which, int64_t where, int64_t skip) noexcept {
    if (length_ == reserved_) {
      return false;
    }
    which_[length_] = which;
    where_[length_] = where;
    skip_[length_] = skip;
    length_++;
    return true;
  }

  inline void pop() noexcept {
    length_--;
  }

  inline int64_t& which() noexcept {
    return which_[length_ - 1];
  }

  inline int64_t& where() noexcept {
    return where_[length_ - 1];
  }

  inline int64_t& skip() noexcept {
    return skip_[length_ - 1];
  }

private:
  int64_t* which_;
  int64_t* where_;
  int64_t* skip_;
  int64_t length_;
  int64_t reserved_;
};


// Instruction values are preprocessor macros for type-ambiguity.

// parser flags (parsers are combined bitwise and then bit-inverted to be negative)
#define PARSER_DIRECT 1
#define PARSER_REPEATED 2
#define PARSER_BIGENDIAN 4
#define PARSER_MASK ~(1 + 2 + 4)
// parser sequential values (starting in the fourth bit)
#define PARSER_BOOL 8
#define PARSER_INT8 16
#define PARSER_INT16 24
#define PARSER_INT32 32
#define PARSER_INT64 40
#define PARSER_INTP 48
#define PARSER_UINT8 56
#define PARSER_UINT16 64
#define PARSER_UINT32 72
#define PARSER_UINT64 80
#define PARSER_UINTP 88
#define PARSER_FLOAT32 96
#define PARSER_FLOAT64 104

// instructions from special parsing rules
#define LITERAL 0
#define IF 1
#define IF_ELSE 2
#define DO 3
#define DO_STEP 4
#define AGAIN 5
#define UNTIL 6
#define WHILE 7
#define EXIT 8
#define PUT 9
#define INC 10
#define GET 11
#define SEEK 12
#define SKIP 13
#define END 14
#define POS 15
#define LEN_INPUT 16
#define REWIND 17
#define LEN_OUTPUT 18
#define WRITE 19
// generic builtin instructions
#define INDEX_I 20
#define INDEX_J 21
#define INDEX_K 22
#define DUP 23
#define DROP 24
#define SWAP 25
#define OVER 26
#define ROT 27
#define NIP 28
#define TUCK 29
#define ADD 30
#define SUB 31
#define MUL 32
#define DIV 33
#define MOD 34
#define DIVMOD 35
#define NEGATE 36
#define ADD1 37
#define SUB1 38
#define ABS 39
#define MIN 40
#define MAX 41
#define EQ 42
#define NE 43
#define GT 44
#define GE 45
#define LT 46
#define LE 47
#define EQ0 48
#define INVERT 49
#define AND 50
#define OR 51
#define XOR 52
#define LSHIFT 53
#define RSHIFT 54
#define FALSE 55
#define TRUE 56
// beginning of the user-defined dictionary
#define DICTIONARY 57

const std::set<std::string> reserved_words_({
  // comments
  "(", ")", "\\", "\n", "",
  // defining functinos
  ":", ";", "recurse",
  // declaring globals
  "variable", "input", "output",
  // conditionals
  "if", "then", "else",
  // loops
  "do", "loop", "+loop",
  "begin", "until", "again", "while", "repeat",
  // nonlocal exits
  "exit",
  // variable access
  "!", "+!", "@",
  // input actions
  "seek", "skip", "end", "pos", "len",
  // output actions
  "rewind", "len", "<-"
});

const std::set<std::string> input_parser_words_({
  "?->", "b->", "h->", "i->", "q->", "n->", "B->", "H->", "I->", "Q->", "N->", "f->", "d->",
  "!h->", "!i->", "!q->", "!n->", "!H->", "!I->", "!Q->", "!N->", "!f->", "!d->",
  "#?->", "#b->", "#h->", "#i->", "#q->", "#n->", "#B->", "#H->", "#I->", "#Q->", "#N->", "#f->", "#d->",
  "#!h->", "#!i->", "#!q->", "#!n->", "#!H->", "#!I->", "#!Q->", "#!N->", "#!f->", "#!d->",
});

const std::map<std::string, int64_t> output_dtype_words_({
  {"bool", PARSER_BOOL},
  {"int8", PARSER_INT8}, {"int16", PARSER_INT16}, {"int32", PARSER_INT32}, {"int64", PARSER_INT64},
  {"uint8", PARSER_UINT8}, {"uint16", PARSER_UINT16}, {"uint32", PARSER_UINT32}, {"uint64", PARSER_UINT64},
  {"float32", PARSER_FLOAT32}, {"float64", PARSER_FLOAT64}
});

const std::map<std::string, int64_t> generic_builtin_words_({
  // loop variables
  {"i", INDEX_I}, {"j", INDEX_J}, {"k", INDEX_K},
  // stack operations
  {"dup", DUP}, {"drop", DROP}, {"swap", SWAP}, {"over", OVER}, {"rot", ROT},
  {"nip", NIP}, {"tuck", TUCK},
  // basic mathematical functions
  {"+", ADD}, {"-", SUB}, {"*", MUL}, {"/", DIV}, {"mod", MOD}, {"/mod", DIVMOD},
  {"negate", NEGATE}, {"1+", ADD1}, {"1-", SUB1}, {"abs", ABS}, {"min", MIN}, {"max", MAX},
  // comparisons
  {"=", EQ}, {"<>", NE}, {">", GT}, {">=", GE}, {"<", LT}, {"<=", LE}, {"0=", EQ0},
  // bitwise operations
  {"invert", INVERT}, {"and", AND}, {"or", OR}, {"xor", XOR},
  {"lshift", LSHIFT}, {"rshift", RSHIFT},
  // constants
  {"false", FALSE}, {"true", TRUE}
});


template <typename T, typename I, bool DEBUG>
class ForthMachine {
public:
  ForthMachine(const std::string& source,
               int64_t initial_buffer=1024,
               double resize_buffer=1.5,
               int64_t initial_stack=1024,
               double resize_stack=1.5,
               int64_t recursion_depth=1024)
    : initial_buffer_(initial_buffer)
    , resize_buffer_(resize_buffer)
    , stack_(initial_stack, resize_stack)
    , recursion_depth_(recursion_depth) {
    compile(source);
  }

  const std::map<std::string, T> variables() const {
    std::map<std::string, T> out;
    for (int64_t i = 0;  i < variable_names_.size();  i++) {
      out[variable_names_[i]] = variables_[i];
    }
    return out;
  }

  const std::vector<std::string> variable_names() const {
    return variable_names_;
  }

  bool has_variable(const std::string& name) const {
    return std::find(variable_names_.begin(),
                     variable_names_.end(), name) != variable_names_.end();
  }

  T variable(const std::string& name) const {
    for (size_t i = 0;  i < variable_names_.size();  i++) {
      if (variable_names_[i] == name) {
        return variables_[i];
      }
    }
    throw std::invalid_argument(
      std::string("unrecognized variable name: ") + name
    );
  }

  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>> run() {
    std::map<std::string, std::shared_ptr<ForthInputBuffer>> inputs;
    std::set<ForthError> ignore;
    return run(inputs, ignore);
  }

  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>> run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs) {
    std::set<ForthError> ignore;
    return run(inputs, ignore);
  }

  const std::map<std::string, std::shared_ptr<ForthOutputBuffer>> run(
      const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs,
      const std::set<ForthError>& ignore) {

    std::vector<std::shared_ptr<ForthInputBuffer>> ins;
    for (auto name : input_names_) {
      auto it = inputs.find(name);
      if (it == inputs.end()) {
        throw std::invalid_argument(
          std::string("name missing from inputs: ") + name
        );
      }
      ins.push_back(it->second);
    }

    std::map<std::string, std::shared_ptr<ForthOutputBuffer>> outputs;
    std::vector<std::shared_ptr<ForthOutputBuffer>> outs;
    for (int64_t i = 0;  i < output_names_.size();  i++) {
      std::string name = output_names_[i];
      std::shared_ptr<ForthOutputBuffer> out;
      switch (output_dtypes_[i]) {
        case dtype::boolean: {
          out = std::make_shared<ForthOutputBufferOf<bool>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int8: {
          out = std::make_shared<ForthOutputBufferOf<int8_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int16: {
          out = std::make_shared<ForthOutputBufferOf<int16_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int32: {
          out = std::make_shared<ForthOutputBufferOf<int32_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::int64: {
          out = std::make_shared<ForthOutputBufferOf<int64_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint8: {
          out = std::make_shared<ForthOutputBufferOf<uint8_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint16: {
          out = std::make_shared<ForthOutputBufferOf<uint16_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint32: {
          out = std::make_shared<ForthOutputBufferOf<uint32_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::uint64: {
          out = std::make_shared<ForthOutputBufferOf<uint64_t>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        // case dtype::float16: { }
        case dtype::float32: {
          out = std::make_shared<ForthOutputBufferOf<float>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        case dtype::float64: {
          out = std::make_shared<ForthOutputBufferOf<double>>(
                    initial_buffer_, resize_buffer_);
          break;
        }
        // case dtype::float128: { }
        // case dtype::complex64: { }
        // case dtype::complex128: { }
        // case dtype::complex256: { }
        // case dtype::datetime64: { }
        // case dtype::timedelta64: { }
        default: {
          throw std::runtime_error("unimplemented ForthOutputBuffer type");
        }
      }
      outputs[name] = out;
      outs.push_back(out);
    }

    stack_.clear();
    for (int64_t i = 0;  i < variables_.size();  i++) {
      variables_[i] = 0;
    }

    ForthError err = ForthError::none;

    do_run(ins, outs, err);

    if (ignore.count(err) == 0) {
      switch (err) {
        case ForthError::stack_underflow: {
          throw std::invalid_argument(
            "in Awkward Forth runtime, stack underflow while filling array");
        }
        case ForthError::read_beyond: {
          throw std::invalid_argument(
            "in Awkward Forth runtime, read beyond end of input while filling array");
        }
        case ForthError::seek_beyond: {
          throw std::invalid_argument(
            "in Awkward Forth runtime, seek beyond input while filling array");
        }
        case ForthError::skip_beyond: {
          throw std::invalid_argument(
            "in Awkward Forth runtime, skip beyond input while filling array");
        }
        case ForthError::rewind_beyond: {
          throw std::invalid_argument(
            "in Awkward Forth runtime, rewind beyond beginning of output while filling array");
        }
      }
    }

    return outputs;
  }

  const std::string tostring() {
    std::stringstream out;
    out << "Variables:" << std::endl;
    for (int64_t i = 0;  i < variable_names_.size();  i++) {
      out << "    " << variable_names_[i] << ": " << variables_[i] << std::endl;
    }
    out << "Stack:" << std::endl << "    " << stack_.tostring() << std::endl;
    return out.str();
  }

  const std::string tostring(
        const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>& outputs) {
    std::stringstream out;
    out << tostring();
    out << "Outputs:" << std::endl;
    for (auto pair : outputs) {
      out << "    " << pair.first << ": " << pair.second.get()->tostring() << std::endl;
    }
    return out.str();
  }

private:
  bool is_integer(const std::string& word, int64_t& value) {
    if (word.size() >= 2  &&  word.substr(0, 2) == std::string("0x")) {
      try {
        value = std::stoul(word.substr(2, (int64_t)word.size() - 2), nullptr, 16);
      }
      catch (std::invalid_argument err) {
        return false;
      }
      return true;
    }
    else {
      try {
        value = std::stoul(word, nullptr, 10);
      }
      catch (std::invalid_argument err) {
        return false;
      }
      return true;
    }
  }

  bool is_variable(const std::string& word) {
    return std::find(variable_names_.begin(),
                     variable_names_.end(), word) != variable_names_.end();
  }

  bool is_input(const std::string& word) {
    return std::find(input_names_.begin(),
                     input_names_.end(), word) != input_names_.end();
  }

  bool is_output(const std::string& word) {
    return std::find(output_names_.begin(),
                     output_names_.end(), word) != output_names_.end();
  }

  bool is_reserved(const std::string& word) {
    return reserved_words_.find(word) != reserved_words_.end()  ||
           input_parser_words_.find(word) != input_parser_words_.end()  ||
           output_dtype_words_.find(word) != output_dtype_words_.end()  ||
           generic_builtin_words_.find(word) != generic_builtin_words_.end();
  }

  void compile(const std::string& source) {
    // // Convert the source code into a list of tokens.
    // std::vector<std::string> tokenized;
    // std::vector<std::pair<int64_t, int64_t>> linecol;
    // int64_t start = 0;
    // int64_t stop = 0;
    // bool full = false;
    // int64_t line = 1;
    // int64_t colstart = 0;
    // int64_t colstop = 0;
    // while (stop < source.size()) {
    //   char current = source[stop];
    //   // Whitespace separates tokens and is not included in them.
    //   if (current == ' '  ||  current == '\r'  ||  current == '\t'  ||
    //       current == '\v'  ||  current == '\f') {
    //     if (full) {
    //       tokenized.push_back(source.substr(start, stop - start));
    //       linecol.emplace_back(line, colstart);
    //     }
    //     start = stop;
    //     full = false;
    //     colstart = colstop;
    //   }
    //   // '\n' is considered a token because it terminates '\\ .. \n' comments.
    //   // It has no semantic meaning after the parsing stage.
    //   else if (current == '\n') {
    //     if (full) {
    //       tokenized.push_back(source.substr(start, stop - start));
    //       linecol.emplace_back(line, colstart);
    //     }
    //     tokenized.push_back(source.substr(stop, 1));
    //     linecol.emplace_back(line, colstart);
    //     start = stop;
    //     full = false;
    //     line += 1;
    //     colstart = 0;
    //     colstop = 0;
    //   }
    //   // Everything else is part of a token (Forth word).
    //   else {
    //     if (!full) {
    //       start = stop;
    //       colstart = colstop;
    //     }
    //     full = true;
    //   }
    //   stop++;
    //   colstop++;
    // }
    // // The source code might end on non-whitespace.
    // if (full) {
    //   tokenized.push_back(source.substr(start, stop - start));
    //   linecol.emplace_back(line, colstart);
    // }

    std::vector<I> instructions;
    std::map<std::string, int64_t> dictionary_names;
    std::vector<std::vector<I>> dictionary;

    // parse("",
    //       tokenized,
    //       linecol,
    //       0,
    //       tokenized.size(),
    //       instructions,
    //       dictionary_names,
    //       dictionary,
    //       0,
    //       0);

    instructions_offsets_.push_back(0);

    // for (auto instruction : instructions) {
    //   instructions_.push_back(instruction);
    // }
    // instructions_offsets_.push_back(instructions_.size());

    // for (auto sequence : dictionary) {
    //   for (auto instruction : sequence) {
    //     instructions_.push_back(instruction);
    //   }
    //   instructions_offsets_.push_back(instructions_.size());
    // }



    input_names_.push_back("testin");
    output_names_.push_back("testout");
    output_dtypes_.push_back(dtype::int32);

    instructions_.push_back(DICTIONARY + 0);
    instructions_.push_back(AGAIN);

    instructions_offsets_.push_back(instructions_.size());

    instructions_.push_back(~(PARSER_INT32 | PARSER_DIRECT));
    instructions_.push_back(0);
    instructions_.push_back(0);

    // instructions_.push_back(LITERAL);
    // instructions_.push_back(10);
    // instructions_.push_back(ADD);

    // instructions_.push_back(WRITE);
    // instructions_.push_back(0);

    instructions_offsets_.push_back(instructions_.size());




    // std::cout << "Instructions offsets:";
    // for (auto x : instructions_offsets_) {
    //   std::cout << " " << x;
    // }
    // std::cout << std::endl;

    // std::cout << "Instructions:";
    // for (auto x : instructions_) {
    //   std::cout << " " << x;
    // }
    // std::cout << std::endl;
  }

  const std::string err_linecol(
      const std::vector<std::pair<int64_t, int64_t>>& linecol, int64_t pos) {
    std::pair<int64_t, int64_t> lc = linecol[pos];
    std::stringstream out;
    out << "in Awkward Forth source code, line " << lc.first << " col " << lc.second << ", ";
    return out.str();
  }

  void parse(const std::string& defn,
             const std::vector<std::string>& tokenized,
             const std::vector<std::pair<int64_t, int64_t>>& linecol,
             int64_t start,
             int64_t stop,
             std::vector<I>& instructions,
             std::map<std::string, int64_t>& dictionary_names,
             std::vector<std::vector<I>>& dictionary,
             int64_t exitdepth,
             int64_t dodepth) {
    int64_t pos = start;
    while (pos < stop) {
      std::string word = tokenized[pos];

      if (word == "(") {
        throw std::runtime_error("not implemented: (");
      }

      else if (word == "\\") {
        throw std::runtime_error("not implemented: \\");
      }

      else if (word == "\n") {
        throw std::runtime_error("not implemented: \n");
      }

      else if (word == "") {
        throw std::runtime_error("not implemented: ");
      }

      else if (word == ":") {
        throw std::runtime_error("not implemented: :");
      }

      else if (word == "recurse") {
        throw std::runtime_error("not implemented: recurse");
      }

      else if (word == "variable") {
        throw std::runtime_error("not implemented: variable");
      }

      else if (word == "input") {
        throw std::runtime_error("not implemented: input");
      }

      else if (word == "output") {
        throw std::runtime_error("not implemented: output");
      }

      else if (word == "if") {
        throw std::runtime_error("not implemented: if");
      }

      else if (word == "do") {
        throw std::runtime_error("not implemented: do");
      }

      else if (word == "begin") {
        throw std::runtime_error("not implemented: begin");
      }

      else if (word == "exit") {
        throw std::runtime_error("not implemented: exit");
      }

      else if (is_variable(word)) {
        throw std::runtime_error("not implemented: is_variable(word)");
      }

      else if (is_input(word)) {
        throw std::runtime_error("not implemented: is_input(word)");
      }

      else if (is_output(word)) {
        throw std::runtime_error("not implemented: is_output(word)");
      }

      else {
        auto generic_builtin = generic_builtin_words_.find(word);
        if (generic_builtin != generic_builtin_words_.end()) {
          throw std::runtime_error("not implemented: is_generic_builtin(word)");
        }

        else {
          auto pair = dictionary_names.find(word);
          if (pair != dictionary_names.end()) {
            throw std::runtime_error("not implemented: is_user_defined(word)");
          }

          else {
            int64_t num;
            if (is_integer(word, num)) {
              instructions_.push_back(LITERAL);
              instructions_.push_back(num);
              pos++;
            }

            else {
              throw std::invalid_argument(
                err_linecol(linecol, pos) +
                std::string("unrecognized word or wrong context for word: ") +
                word
              );
            }
          }
        }
      }
    }
  }

  inline I get_instruction(ForthInstructionPointer& pointer) noexcept {
    int64_t start = instructions_offsets_[pointer.which()];
    return instructions_[start + pointer.where()];
  }

  inline void write_from_stack(const std::vector<std::shared_ptr<ForthOutputBuffer>>& outs,
                               int64_t num,
                               T* top) noexcept;

  void do_run(const std::vector<std::shared_ptr<ForthInputBuffer>>& ins,
              const std::vector<std::shared_ptr<ForthOutputBuffer>>& outs,
              ForthError& err) noexcept {
    ForthInstructionPointer pointer(recursion_depth_);
    pointer.push(0, 0, 0);

    while (!pointer.empty()) {
      while (pointer.where() < (instructions_offsets_[pointer.which() + 1] -
                                instructions_offsets_[pointer.which()])) {
        I instruction = get_instruction(pointer);
        pointer.where() += 1;

        if (instruction < 0) {
          bool byteswap;
          if (NATIVELY_BIG_ENDIAN) {
            byteswap = ((~instruction & PARSER_BIGENDIAN) == 0);
          }
          else {
            byteswap = ((~instruction & PARSER_BIGENDIAN) != 0);
          }

          I in_num = get_instruction(pointer);
          pointer.where() += 1;

          int64_t num_items = 1;
          if (~instruction & PARSER_REPEATED) {
            num_items = stack_.pop(err);
            if (err != ForthError::none) {
              return;
            }
          }

          if (~instruction & PARSER_DIRECT) {
            I out_num = get_instruction(pointer);
            pointer.where() += 1;

            switch (~instruction & PARSER_MASK) {
              case PARSER_BOOL: {
                bool* ptr = reinterpret_cast<bool*>(
                    ins[in_num].get()->read(num_items * sizeof(bool), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_bool(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_bool(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_INT8: {
                int8_t* ptr = reinterpret_cast<int8_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int8_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_int8(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_int8(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_INT16: {
                int16_t* ptr = reinterpret_cast<int16_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int16_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_int16(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_int16(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_INT32: {
                int32_t* ptr = reinterpret_cast<int32_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int32_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_int32(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_int32(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_INT64: {
                int64_t* ptr = reinterpret_cast<int64_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int64_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_int64(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_int64(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_INTP: {
                ssize_t* ptr = reinterpret_cast<ssize_t*>(
                    ins[in_num].get()->read(num_items * sizeof(ssize_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_intp(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_intp(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_UINT8: {
                uint8_t* ptr = reinterpret_cast<uint8_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint8_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_uint8(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_uint8(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_UINT16: {
                uint16_t* ptr = reinterpret_cast<uint16_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint16_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_uint16(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_uint16(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_UINT32: {
                uint32_t* ptr = reinterpret_cast<uint32_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint32_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_uint32(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_uint32(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_UINT64: {
                uint64_t* ptr = reinterpret_cast<uint64_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint64_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_uint64(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_uint64(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_UINTP: {
                size_t* ptr = reinterpret_cast<size_t*>(
                    ins[in_num].get()->read(num_items * sizeof(size_t), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_uintp(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_uintp(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_FLOAT32: {
                float* ptr = reinterpret_cast<float*>(
                    ins[in_num].get()->read(num_items * sizeof(float), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_float32(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_float32(num_items, ptr, byteswap);
                }
                break;
              }

              case PARSER_FLOAT64: {
                double* ptr = reinterpret_cast<double*>(
                    ins[in_num].get()->read(num_items * sizeof(double), err));
                if (err != ForthError::none) {
                  return;
                }
                if (num_items == 1) {
                  outs[out_num].get()->write_one_float64(*ptr, byteswap);
                }
                else {
                  outs[out_num].get()->write_float64(num_items, ptr, byteswap);
                }
                break;
              }
            }
          }
          else {
            switch (~instruction & PARSER_MASK) {
              case PARSER_BOOL: {
                bool* ptr = reinterpret_cast<bool*>(
                    ins[in_num].get()->read(num_items * sizeof(bool), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  bool value = ptr[i];
                  stack_.push(value);
                }
                break;
              }

              case PARSER_INT8: {
                int8_t* ptr = reinterpret_cast<int8_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int8_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  int8_t value = ptr[i];
                  stack_.push(value);
                }
                break;
              }

              case PARSER_INT16: {
                int16_t* ptr = reinterpret_cast<int16_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int16_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  int16_t value = ptr[i];
                  if (byteswap) {
                    byteswap16(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_INT32: {
                int32_t* ptr = reinterpret_cast<int32_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int32_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  int32_t value = ptr[i];
                  if (byteswap) {
                    byteswap32(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_INT64: {
                int64_t* ptr = reinterpret_cast<int64_t*>(
                    ins[in_num].get()->read(num_items * sizeof(int64_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  int64_t value = ptr[i];
                  if (byteswap) {
                    byteswap64(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_INTP: {
                ssize_t* ptr = reinterpret_cast<ssize_t*>(
                    ins[in_num].get()->read(num_items * sizeof(ssize_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  ssize_t value = ptr[i];
                  if (byteswap) {
                    if (sizeof(ssize_t) == 4) {
                      byteswap32(1, &value);
                    }
                    else {
                      byteswap64(1, &value);
                    }
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_UINT8: {
                uint8_t* ptr = reinterpret_cast<uint8_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint8_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  uint8_t value = ptr[i];
                  stack_.push(value);
                }
                break;
              }

              case PARSER_UINT16: {
                uint16_t* ptr = reinterpret_cast<uint16_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint16_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  uint16_t value = ptr[i];
                  if (byteswap) {
                    byteswap16(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_UINT32: {
                uint32_t* ptr = reinterpret_cast<uint32_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint32_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  uint32_t value = ptr[i];
                  if (byteswap) {
                    byteswap32(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_UINT64: {
                uint64_t* ptr = reinterpret_cast<uint64_t*>(
                    ins[in_num].get()->read(num_items * sizeof(uint64_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  uint64_t value = ptr[i];
                  if (byteswap) {
                    byteswap64(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_UINTP: {
                size_t* ptr = reinterpret_cast<size_t*>(
                    ins[in_num].get()->read(num_items * sizeof(size_t), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  size_t value = ptr[i];
                  if (byteswap) {
                    if (sizeof(size_t) == 4) {
                      byteswap32(1, &value);
                    }
                    else {
                      byteswap64(1, &value);
                    }
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_FLOAT32: {
                float* ptr = reinterpret_cast<float*>(
                    ins[in_num].get()->read(num_items * sizeof(float), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  float value = ptr[i];
                  if (byteswap) {
                    byteswap32(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }

              case PARSER_FLOAT64: {
                double* ptr = reinterpret_cast<double*>(
                    ins[in_num].get()->read(num_items * sizeof(double), err));
                if (err != ForthError::none) {
                  return;
                }
                for (int64_t i = 0;  i < num_items;  i++) {
                  double value = ptr[i];
                  if (byteswap) {
                    byteswap64(1, &value);
                  }
                  stack_.push(value);
                }
                break;
              }
            }
          }
        }

        else if (instruction >= DICTIONARY) {
          pointer.push((instruction - DICTIONARY) + 1, 0, 0);
        }

        else {
          switch (instruction) {
            case LITERAL: {
              I num = get_instruction(pointer);
              pointer.where() += 1;
              stack_.push((T)num);
              break;
            }

            case PUT: {
              I num = get_instruction(pointer);
              pointer.where() += 1;
              T value = stack_.pop(err);
              if (err != ForthError::none) {
                return;
              }
              variables_[num] = value;
              break;
            }

            case INC: {
              I num = get_instruction(pointer);
              pointer.where() += 1;
              T value = stack_.pop(err);
              if (err != ForthError::none) {
                return;
              }
              variables_[num] += value;
              break;
            }

            case GET: {
              I num = get_instruction(pointer);
              pointer.where() += 1;
              stack_.push(variables_[num]);
              break;
            }

            case SKIP: {
              break;
            }

            case SEEK: {
              break;
            }

            case END: {
              break;
            }

            case POS: {
              break;
            }

            case LEN_INPUT: {
              break;
            }

            case REWIND: {
              break;
            }

            case LEN_OUTPUT: {
              break;
            }

            case WRITE: {
              I num = get_instruction(pointer);
              pointer.where() += 1;
              T* top = stack_.peek();
              stack_.pop(err);
              if (err != ForthError::none) {
                return;
              }
              write_from_stack(outs, num, top);
              break;
            }

            case IF: {
              break;
            }

            case IF_ELSE: {
              break;
            }

            case DO: {
              break;
            }

            case DO_STEP: {
              break;
            }

            case AGAIN: {
              // Go back and do the body again.
              pointer.where() -= 2;
              break;
            }

            case UNTIL: {
              break;
            }

            case WHILE: {
              break;
            }

            case EXIT: {
              break;
            }

            case INDEX_I: {
              break;
            }

            case INDEX_J: {
              break;
            }

            case INDEX_K: {
              break;
            }

            case DUP: {
              break;
            }

            case DROP: {
              stack_.drop(err);
              if (err != ForthError::none) {
                return;
              }
              break;
            }

            case SWAP: {
              break;
            }

            case OVER: {
              break;
            }

            case ROT: {
              break;
            }

            case NIP: {
              break;
            }

            case TUCK: {
              break;
            }

            case ADD: {
              T* pair = stack_.pop2(err);
              if (err != ForthError::none) {
                return;
              }
              stack_.push(pair[0] + pair[1]);
              break;
            }

            case SUB: {
              break;
            }

            case MUL: {
              break;
            }

            case DIV: {
              break;
            }

            case MOD: {
              break;
            }

            case DIVMOD: {
              break;
            }

            case LSHIFT: {
              break;
            }

            case RSHIFT: {
              break;
            }

            case ABS: {
              break;
            }

            case MIN: {
              break;
            }

            case MAX: {
              break;
            }

            case NEGATE: {
              break;
            }

            case ADD1: {
              break;
            }

            case SUB1: {
              break;
            }

            case EQ0: {
              break;
            }

            case EQ: {
              break;
            }

            case NE: {
              break;
            }

            case GT: {
              break;
            }

            case GE: {
              break;
            }

            case LT: {
              break;
            }

            case LE: {
              break;
            }

            case AND: {
              break;
            }

            case OR: {
              break;
            }

            case XOR: {
              break;
            }

            case INVERT: {
              break;
            }

            case FALSE: {
              break;
            }

            case TRUE: {
              break;
            }
          }
        } // end handle one instruction

      } // end walk over instructions in this segment

      pointer.pop();
    } // end of all segments
  }

  int64_t initial_buffer_;
  double resize_buffer_;

  ForthStack<T> stack_;
  int64_t recursion_depth_;

  std::vector<std::string> variable_names_;
  std::vector<T> variables_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<dtype> output_dtypes_;

  std::vector<int64_t> instructions_offsets_;
  std::vector<I> instructions_;
};


template <>
void ForthMachine<int32_t, int32_t, true>::write_from_stack(
    const std::vector<std::shared_ptr<ForthOutputBuffer>>& outs,
    int64_t num,
    int32_t* top) noexcept {
  if (num == 1) {
    outs[num].get()->write_one_int32(*top, false);
  }
  else {
    outs[num].get()->write_int32(1, top, false);
  }
}


int main() {
  ForthMachine<int32_t, int32_t, true> vm("");

  const int64_t length = 1000000;
  // const int64_t length = 20;

  std::shared_ptr<int32_t> test_input_ptr = std::shared_ptr<int32_t>(
      new int32_t[length], array_deleter<int32_t>());
  for (int64_t i = 0;  i < length;  i++) {
    test_input_ptr.get()[i] = (i % 9) - 4;
  }

  ForthError err = ForthError::none;
  std::map<std::string, std::shared_ptr<ForthInputBuffer>> inputs;
  inputs["testin"] = std::make_shared<ForthInputBuffer>(test_input_ptr,
                                                        0,
                                                        sizeof(int32_t) * length);

  for (int64_t repeat = 0;  repeat < 4;  repeat++) {
    std::vector<std::shared_ptr<ForthInputBuffer>> ins({ inputs["testin"] });
    std::vector<std::shared_ptr<ForthOutputBuffer>> outs({
        std::make_shared<ForthOutputBufferOf<int64_t>>() });

    auto cpp_begin = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0;  i < length;  i += 1) {
      int32_t* ptr = reinterpret_cast<int32_t*>(ins[0].get()->read(sizeof(int32_t) * 1, err));
      outs[0].get()->write_one_int32(*ptr, false);
    }
    auto cpp_end = std::chrono::high_resolution_clock::now();

    std::cout << "                       C++ time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(cpp_end - cpp_begin).count()
              << " us" << std::endl;

    inputs["testin"].get()->seek(0, err);
  }

  for (int64_t repeat = 0;  repeat < 4;  repeat++) {
    std::set<ForthError> ignore({ ForthError::read_beyond });

    auto forth_begin = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::shared_ptr<ForthOutputBuffer>> outputs = vm.run(inputs, ignore);
    auto forth_end = std::chrono::high_resolution_clock::now();

    // std::cout << vm.tostring(outputs);
    std::cout << "Forth time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(forth_end - forth_begin).count()
              << " us" << std::endl;

    inputs["testin"].get()->seek(0, err);
  }

}
