// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYBUILDEROPTIONS_H_
#define AWKWARD_ARRAYBUILDEROPTIONS_H_

#include <cmath>
#include <cstring>

#include "awkward/common.h"

namespace awkward {
  /// @class ArrayBuilderOptions
  ///
  /// @brief Container for all configuration options needed by ArrayBuilder,
  /// GrowableBuffer, and the Builder subclasses.
  class LIBAWKWARD_EXPORT_SYMBOL ArrayBuilderOptions {
  public:
    /// @brief Creates an ArrayBuilderOptions from a full set of parameters.
    ///
    /// @param initial The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    /// @param resize The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    ArrayBuilderOptions(int64_t initial, double resize,
      bool convert_nan_and_inf = false, bool replace_nan_and_inf_ = false);

    /// @brief The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    int64_t
      initial() const;

    /// @brief The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    double
      resize() const;

    /// @brief Configurable option flag to convert 'NaN' and 'Inf' to floats
    bool
      convertNanAndInf() const { return convert_nan_and_inf_; }

    /// @brief Configurable option flag to replace 'NaN' and 'Inf' with
    /// alternative configurable strings
    bool
      replaceNanAndInf() const { return replace_nan_and_inf_; }

  private:
    /// See #initial.
    int64_t initial_;
    /// See #resize.
    double resize_;
    /// @brief Flag to convert 'NaN' and 'Inf' to floats
    const bool convert_nan_and_inf_;
    /// @brief Flag to replace 'NaN' and 'Inf' with strings
    const bool replace_nan_and_inf_;
  };
}

#endif // AWKWARD_ARRAYBUILDEROPTIONS_H_
