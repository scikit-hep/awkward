// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_ARRAYBUILDEROPTIONS_H_
#define AWKWARD_ARRAYBUILDEROPTIONS_H_

namespace awkward {
  /// @class ArrayBuilderOptions
  ///
  /// @brief Container for all configuration options needed by ArrayBuilder,
  /// GrowableBuffer, and the Builder subclasses.
  struct ArrayBuilderOptions {
    /// @brief Creates an ArrayBuilderOptions from a full set of parameters.
    ///
    /// @param initial The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    /// @param resize The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    ArrayBuilderOptions(int64_t from_initial, double from_resize)
      : initial(from_initial) ,
        resize(from_resize) {}

    /// @brief The initial number of
    /// {@link GrowableBuffer#reserved reserved} entries for a GrowableBuffer.
    int64_t initial;

    /// @brief The factor with which a GrowableBuffer is resized
    /// when its {@link GrowableBuffer#length length} reaches its
    /// {@link GrowableBuffer#reserved reserved}.
    double resize;
  };
}

#endif // AWKWARD_ARRAYBUILDEROPTIONS_H_
