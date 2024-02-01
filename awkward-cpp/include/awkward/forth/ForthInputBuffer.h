// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_FORTHINPUTBUFFER_H_
#define AWKWARD_FORTHINPUTBUFFER_H_

#include <memory>

#include "awkward/common.h"
#include "awkward/util.h"

namespace awkward {
  /// @class ForthInputBuffer
  ///
  /// @brief HERE
  ///
  /// THERE
  class EXPORT_SYMBOL ForthInputBuffer {
  public:
    /// @brief HERE
    ForthInputBuffer(const std::shared_ptr<void> ptr,
                     int64_t offset,
                     int64_t length);

    /// @brief HERE
    uint8_t
      peek_byte(int64_t after, util::ForthError& err) noexcept;

    /// @brief HERE
    void*
      read(int64_t num_bytes, util::ForthError& err) noexcept;

    /// @brief HERE
    uint8_t
      read_byte(util::ForthError& err) noexcept;

    /// @brief HERE
    int64_t
      read_enum(const std::vector<std::string>& strings, int64_t start, int64_t stop) noexcept;

    /// @brief HERE
    uint64_t
      read_varint(util::ForthError& err) noexcept;

    /// @brief HERE
    int64_t
      read_zigzag(util::ForthError& err) noexcept;

    /// @brief HERE
    int64_t
      read_textint(util::ForthError& err) noexcept;

    /// @brief HERE
    double
      read_textfloat(util::ForthError& err) noexcept;

    /// @brief HERE
    void
      read_quotedstr(char* string_buffer, int64_t max_string_size, int64_t& length,
                     util::ForthError& err) noexcept;

    /// @brief HERE
    void
      seek(int64_t to, util::ForthError& err) noexcept;

    /// @brief HERE
    void
      skip(int64_t num_bytes, util::ForthError& err) noexcept;

    /// @brief HERE
    void
      skipws() noexcept;

    /// @brief HERE
    bool
      end() const noexcept;

    /// @brief HERE
    int64_t
      pos() const noexcept;

    /// @brief HERE
    int64_t
      len() const noexcept;

    /// @brief Returns a shared pointer to an AwkwardForth input buffer.
    std::shared_ptr<void>
      ptr() noexcept;

  private:
    std::shared_ptr<void> ptr_;
    int64_t offset_;
    int64_t length_;
    int64_t pos_;
  };
}

#endif // AWKWARD_FORTHINPUTBUFFER_H_
