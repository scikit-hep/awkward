// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYBUILDEROPTIONS_H_
#define AWKWARD_ARRAYBUILDEROPTIONS_H_

#include <cmath>
#include <cstring>

#include "awkward/cpu-kernels/util.h"

namespace awkward {
  /// @class ArrayBuilderOptions
  ///
  /// @brief Container for all configuration options needed by ArrayBuilder,
  /// GrowableBuffer, and the Builder subclasses.
  class EXPORT_SYMBOL EXPORT_TYPE ArrayBuilderOptions {
  public:
    /// @brief Creates an ArrayBuilderOptions from a full set of parameters.
    ///
    /// @param initial The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    /// @param resize The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    ArrayBuilderOptions(int64_t initial, double resize);

    /// @brief The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    int64_t
      initial() const;

    /// @brief The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    double
      resize() const;

  private:
    /// See #initial.
    int64_t initial_;
    /// See #resize.
    double resize_;
  };
}

#endif // AWKWARD_ARRAYBUILDEROPTIONS_H_
