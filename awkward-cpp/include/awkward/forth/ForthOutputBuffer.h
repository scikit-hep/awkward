// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_FORTHOUTPUTBUFFER_H_
#define AWKWARD_FORTHOUTPUTBUFFER_H_

#include "awkward/common.h"
#include "awkward/util.h"

namespace awkward {
  #define NATIVELY_BIG_ENDIAN (*(std::uint16_t *)"\0\xff" < 0x100)

  /// @brief HERE
  template <typename T>
  void byteswap16(std::int64_t num_items, T& value);

  /// @brief HERE
  template <typename T>
  void byteswap32(std::int64_t num_items, T& value);

  /// @brief HERE
  template <typename T>
  void byteswap64(std::int64_t num_items, T& value);

  /// @brief HERE
  template <typename T>
  void byteswap_intp(std::int64_t num_items, T& value);

  /// @class ForthOutputBuffer
  ///
  /// @brief HERE
  ///
  /// THERE
  class EXPORT_SYMBOL ForthOutputBuffer {
  public:
    ForthOutputBuffer(std::int64_t initial, double resize);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~ForthOutputBuffer();

    /// @brief HERE
    std::int64_t
      len() const noexcept;

    /// @brief HERE
    void
      rewind(std::int64_t num_items, util::ForthError& err) noexcept;

    /// @brief HERE
    void
      reset() noexcept;

    /// @brief HERE
    virtual void
      dup(std::int64_t num_times, util::ForthError& err) noexcept = 0;

    /// @brief HERE
    virtual const std::shared_ptr<void>
      ptr() const noexcept = 0;

    virtual util::dtype
      dtype() const = 0;

    /// @brief HERE
    virtual void
      write_one_bool(bool value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_int8(std::int8_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_int16(std::int16_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_int32(std::int32_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_int64(std::int64_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_intp(ssize_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_uint8(std::uint8_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_uint16(std::uint16_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_uint32(std::uint32_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_uint64(std::uint64_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_uintp(std::size_t value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_float32(float value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_float64(double value, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_one_string(char* string_buffer, int64_t length) noexcept = 0;

    /// @brief HERE
    virtual void
      write_bool(std::int64_t num_items, bool* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_int8(std::int64_t num_items, std::int8_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_int16(std::int64_t num_items, std::int16_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_int32(std::int64_t num_items, std::int32_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_int64(std::int64_t num_items, std::int64_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_intp(std::int64_t num_items, ssize_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_const_uint8(std::int64_t num_items, const std::uint8_t* values) noexcept = 0;

    /// @brief HERE
    virtual void
      write_uint8(std::int64_t num_items, std::uint8_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_uint16(std::int64_t num_items, std::uint16_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_uint32(std::int64_t num_items, std::uint32_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_uint64(std::int64_t num_items, std::uint64_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_uintp(std::int64_t num_items, std::size_t* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_float32(std::int64_t num_items, float* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_float64(std::int64_t num_items, double* values, bool byteswap) noexcept = 0;

    /// @brief HERE
    virtual void
      write_add_int32(std::int32_t value) noexcept = 0;

    /// @brief HERE
    virtual void
      write_add_int64(std::int64_t value) noexcept = 0;

    virtual std::string
      tostring() const = 0;

  protected:
    std::int64_t length_;
    std::int64_t reserved_;
    double resize_;
  };

  template <typename OUT>
  class EXPORT_SYMBOL ForthOutputBufferOf : public ForthOutputBuffer {
  public:
    ForthOutputBufferOf(std::int64_t initial, double resize);

    void
      dup(std::int64_t num_times, util::ForthError& err) noexcept override;

    const std::shared_ptr<void>
      ptr() const noexcept override;

    util::dtype
      dtype() const override;

    void
      write_one_bool(bool value, bool byteswap) noexcept override;

    void
      write_one_int8(int8_t value, bool byteswap) noexcept override;

    void
      write_one_int16(int16_t value, bool byteswap) noexcept override;

    void
      write_one_int32(int32_t value, bool byteswap) noexcept override;

    void
      write_one_int64(int64_t value, bool byteswap) noexcept override;

    void
      write_one_intp(ssize_t value, bool byteswap) noexcept override;

    void
      write_one_uint8(uint8_t value, bool byteswap) noexcept override;

    void
      write_one_uint16(uint16_t value, bool byteswap) noexcept override;

    void
      write_one_uint32(uint32_t value, bool byteswap) noexcept override;

    void
      write_one_uint64(uint64_t value, bool byteswap) noexcept override;

    void
      write_one_uintp(size_t value, bool byteswap) noexcept override;

    void
      write_one_float32(float value, bool byteswap) noexcept override;

    void
      write_one_float64(double value, bool byteswap) noexcept override;

    void
      write_one_string(char* string_buffer, std::int64_t length) noexcept override;

    void
      write_bool(std::int64_t num_items, bool* values, bool byteswap) noexcept override;

    void
      write_int8(std::int64_t num_items, std::int8_t* values, bool byteswap) noexcept override;

    void
      write_int16(std::int64_t num_items, std::int16_t* values, bool byteswap) noexcept override;

    void
      write_int32(std::int64_t num_items, std::int32_t* values, bool byteswap) noexcept override;

    void
      write_int64(std::int64_t num_items, std::int64_t* values, bool byteswap) noexcept override;

    void
      write_intp(std::int64_t num_items, ssize_t* values, bool byteswap) noexcept override;

    void
      write_const_uint8(std::int64_t num_items, const std::uint8_t* values) noexcept override;

    void
      write_uint8(std::int64_t num_items, std::uint8_t* values, bool byteswap) noexcept override;

    void
      write_uint16(std::int64_t num_items, std::uint16_t* values, bool byteswap) noexcept override;

    void
      write_uint32(std::int64_t num_items, std::uint32_t* values, bool byteswap) noexcept override;

    void
      write_uint64(std::int64_t num_items, std::uint64_t* values, bool byteswap) noexcept override;

    void
      write_uintp(std::int64_t num_items, std::size_t* values, bool byteswap) noexcept override;

    void
      write_float32(std::int64_t num_items, float* values, bool byteswap) noexcept override;

    void
      write_float64(std::int64_t num_items, double* values, bool byteswap) noexcept override;

    void
      write_add_int32(std::int32_t value) noexcept override;

    void
      write_add_int64(std::int64_t value) noexcept override;

    std::string tostring() const override;

  private:

    /// @brief HERE
    void
      maybe_resize(std::int64_t next);

    /// @brief HERE
    template <typename IN>
    inline void write_one(IN value) noexcept {
      length_++;
      maybe_resize(length_);
      ptr_.get()[length_ - 1] = (OUT)value;
    }

    /// @brief HERE
    template <typename IN>
    inline void write_copy(std::int64_t num_items, const IN* values) noexcept {
      std::int64_t next = length_ + num_items;
      maybe_resize(next);
      for (std::int64_t i = 0;  i < num_items;  i++) {
        ptr_.get()[length_ + i] = (OUT)values[i];
      }
      length_ = next;
    }

    std::shared_ptr<OUT> ptr_;
  };

}

#endif // AWKWARD_FORTHOUTPUTBUFFER_H_
