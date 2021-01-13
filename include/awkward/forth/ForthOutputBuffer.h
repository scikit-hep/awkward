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

    virtual const std::shared_ptr<void>
      ptr() const noexcept = 0;

  private:
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

  public:
    std::shared_ptr<OUT> ptr_;
  };

}

#endif // AWKWARD_FORTHOUTPUTBUFFER_H_
