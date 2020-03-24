// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_PRIMITIVETYPE_H_
#define AWKWARD_PRIMITIVETYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  /// @class PrimitiveType
  ///
  /// @brief Describes the high level type of an array that contains fixed-size
  /// items, such as numbers or booleans.
  class EXPORT_SYMBOL PrimitiveType: public Type {
  public:
    /// @brief Types that can be described by a PrimitiveType.
    ///
    /// Note that NumpyArray and {@link RawArrayOf RawArray} can hold types
    /// of data that cannot be described by a PrimitiveType.
    enum DType {
      boolean,
      int8,
      int16,
      int32,
      int64,
      uint8,
      uint16,
      uint32,
      uint64,
      float32,
      float64,
      numtypes
    };

    /// Constructs a PrimitiveType with a full set of parameters.
    ///
    /// @param parameters Custom parameters inherited from the Content that
    /// this type describes.
    /// @param typestr Optional string that overrides the default string
    /// representation (missing if empty).
    /// @param dtype The tag that defines this PrimitiveType.
    PrimitiveType(const util::Parameters& parameters,
                  const std::string& typestr, DType dtype);

    /// @brief The tag that defines this PrimitiveType.
    const DType
      dtype() const;

    std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    const TypePtr
      shallow_copy() const override;

    bool
      equal(const TypePtr& other, bool check_parameters) const override;

    int64_t
      numfields() const override;

    int64_t
      fieldindex(const std::string& key) const override;

    const std::string
      key(int64_t fieldindex) const override;

    bool
      haskey(const std::string& key) const override;

    const std::vector<std::string>
      keys() const override;

    const ContentPtr
      empty() const override;

  private:
    const DType dtype_;
  };

}

#endif // AWKWARD_PRIMITIVETYPE_H_
