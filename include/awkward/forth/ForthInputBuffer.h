// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
  class LIBAWKWARD_EXPORT_SYMBOL ForthInputBuffer {
  public:
    /// @brief HERE
    ForthInputBuffer(const std::shared_ptr<void> ptr,
                     int64_t offset,
                     int64_t length);

    /// @brief HERE
    void*
      read(int64_t num_bytes, util::ForthError& err);  // noexcept

    /// @brief HERE
    void
      seek(int64_t to, util::ForthError& err);  // noexcept

    /// @brief HERE
    void
      skip(int64_t num_bytes, util::ForthError& err);  // noexcept

    /// @brief HERE
    bool
      end() const;  // noexcept

    /// @brief HERE
    int64_t
      pos() const;  // noexcept

    /// @brief HERE
    int64_t
      len() const;  // noexcept

  private:
    std::shared_ptr<void> ptr_;
    int64_t offset_;
    int64_t length_;
    int64_t pos_;
  };
}

#endif // AWKWARD_FORTHINPUTBUFFER_H_
