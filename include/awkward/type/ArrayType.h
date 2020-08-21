// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYTYPE_H_
#define AWKWARD_ARRAYTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  /// @class ArrayType
  ///
  /// @brief Describes the high level type of a user-facing array, i.e.
  /// `ak.Array` in Python, as opposed to Content. The #length of the array
  /// is part of its type.
  ///
  /// No Content nodes have this type. The #length makes it non-composable.
  class LIBAWKWARD_EXPORT_SYMBOL ArrayType: public Type {
  public:
    /// @brief Create an ArrayType with a full set of parameters.
    ///
    /// @param parameters Custom parameters (not used).
    /// @param typestr Optional string that overrides the default string
    /// representation (missing if empty).
    /// @param type The Type of the composable Content.
    /// @param length The length of the array.
    ArrayType(const util::Parameters& parameters,
              const std::string& typestr,
              const TypePtr& type,
              int64_t length);

    /// @brief The Type of the composable Content.
    const TypePtr
      type() const;

    /// @brief The length of the array.
    int64_t
      length() const;

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
    /// @brief See #type.
    TypePtr type_;
    /// @brief See #length.
    int64_t length_;
  };
}

#endif // AWKWARD_ARRAYTYPE_H_
