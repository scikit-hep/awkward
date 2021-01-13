// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_FORTHOUTPUTBUFFER_H_
#define AWKWARD_FORTHOUTPUTBUFFER_H_

// #include <cstring>

#include "awkward/common.h"

namespace awkward {
  /// @class ForthOutputBuffer
  ///
  /// @brief HERE
  ///
  /// THERE
  class LIBAWKWARD_EXPORT_SYMBOL ForthOutputBuffer {
  public:
    ForthOutputBuffer(int64_t initial, double resize);

    /// @brief HERE
    int64_t
      len() const;  // noexcept

    /// @brief HERE
    void
      rewind(int64_t num_bytes, util::ForthError& err);  // noexcept

    /// @brief HERE
    virtual const std::shared_ptr<void>
      ptr() const noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_bool(bool value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_int8(int8_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_int16(int16_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_int32(int32_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_int64(int64_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_intp(ssize_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_uint8(uint8_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_uint16(uint16_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_uint32(uint32_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_uint64(uint64_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_uintp(size_t value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_float32(float value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_one_float64(double value, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_bool(int64_t num_items, bool* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_int8(int64_t num_items, int8_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_int16(int64_t num_items, int16_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_int32(int64_t num_items, int32_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_int64(int64_t num_items, int64_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_intp(int64_t num_items, ssize_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_uint8(int64_t num_items, uint8_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_uint16(int64_t num_items, uint16_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_uint32(int64_t num_items, uint32_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_uint64(int64_t num_items, uint64_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_uintp(int64_t num_items, size_t* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_float32(int64_t num_items, float* values, bool byteswap) = 0;  // noexcept

    /// @brief HERE
    virtual void
      write_float64(int64_t num_items, double* values, bool byteswap) = 0;  // noexcept

  protected:
    int64_t length_;
    int64_t reserved_;
    double resize_;
  };

  template <typename OUT>
  class LIBAWKWARD_EXPORT_SYMBOL ForthOutputBufferOf : public ForthOutputBuffer {
  public:
    ForthOutputBufferOf(int64_t initial, double resize);

    const std::shared_ptr<void>
      ptr() const noexcept override;

    void
      write_one_bool(bool value, bool byteswap) override;  // noexcept

    void
      write_one_int8(int8_t value, bool byteswap) override;  // noexcept

    void
      write_one_int16(int16_t value, bool byteswap) override;  // noexcept

    void
      write_one_int32(int32_t value, bool byteswap) override;  // noexcept

    void
      write_one_int64(int64_t value, bool byteswap) override;  // noexcept

    void
      write_one_intp(ssize_t value, bool byteswap) override;  // noexcept

    void
      write_one_uint8(uint8_t value, bool byteswap) override;  // noexcept

    void
      write_one_uint16(uint16_t value, bool byteswap) override;  // noexcept

    void
      write_one_uint32(uint32_t value, bool byteswap) override;  // noexcept

    void
      write_one_uint64(uint64_t value, bool byteswap) override;  // noexcept

    void
      write_one_uintp(size_t value, bool byteswap) override;  // noexcept

    void
      write_one_float32(float value, bool byteswap) override;  // noexcept

    void
      write_one_float64(double value, bool byteswap) override;  // noexcept

    void
      write_bool(int64_t num_items, bool* values, bool byteswap) override;  // noexcept

    void
      write_int8(int64_t num_items, int8_t* values, bool byteswap) override;  // noexcept

    void
      write_int16(int64_t num_items, int16_t* values, bool byteswap) override;  // noexcept

    void
      write_int32(int64_t num_items, int32_t* values, bool byteswap) override;  // noexcept

    void
      write_int64(int64_t num_items, int64_t* values, bool byteswap) override;  // noexcept

    void
      write_intp(int64_t num_items, ssize_t* values, bool byteswap) override;  // noexcept

    void
      write_uint8(int64_t num_items, uint8_t* values, bool byteswap) override;  // noexcept

    void
      write_uint16(int64_t num_items, uint16_t* values, bool byteswap) override;  // noexcept

    void
      write_uint32(int64_t num_items, uint32_t* values, bool byteswap) override;  // noexcept

    void
      write_uint64(int64_t num_items, uint64_t* values, bool byteswap) override;  // noexcept

    void
      write_uintp(int64_t num_items, size_t* values, bool byteswap) override;  // noexcept

    void
      write_float32(int64_t num_items, float* values, bool byteswap) override;  // noexcept

    void
      write_float64(int64_t num_items, double* values, bool byteswap) override;  // noexcept

  public:

    /// @brief HERE
    void
      maybe_resize(int64_t next);

    /// @brief HERE
    template <typename IN>
    inline void write_one(IN value) { // noexcept
      length_++;
      maybe_resize(length_);
      ptr_.get()[length_ - 1] = value;
    }

    /// @brief HERE
    template <typename IN>
    inline void write_copy(int64_t num_items, const IN* values) {  // noexcept
      int64_t next = length_ + num_items;
      maybe_resize(next);
      for (int64_t i = 0;  i < num_items;  i++) {
        ptr_.get()[length_ + i] = values[i];
      }
      length_ = next;
    }

    std::shared_ptr<OUT> ptr_;
  };

}

#endif // AWKWARD_FORTHOUTPUTBUFFER_H_
