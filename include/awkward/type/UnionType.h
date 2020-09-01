// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONTYPE_H_
#define AWKWARD_UNIONTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  /// @class UnionType
  ///
  /// @brief Describes the high level type of heterogeneous data.
  ///
  /// {@link UnionArrayOf UnionArray} nodes have this type.
  class LIBAWKWARD_EXPORT_SYMBOL UnionType: public Type {
  public:
    /// @brief Create an UnionArray with a full set of parameters.
    ///
    /// @param parameters Custom parameters inherited from the Content that
    /// this type describes.
    /// @param typestr Optional string that overrides the default string
    /// representation (missing if empty).
    /// @param types The Type of each possibility.
    UnionType(const util::Parameters& parameters,
              const std::string& typestr,
              const std::vector<TypePtr>& types);

    /// @brief The Type of each possibility.
    const std::vector<TypePtr>
      types() const;

    /// @brief The number of possible types.
    int64_t
      numtypes() const;

    /// @brief Returns the type at a given index.
    const TypePtr
      type(int64_t index) const;

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
    /// @brief See #types.
    const std::vector<TypePtr> types_;
  };
}

#endif // AWKWARD_OPTIONTYPE_H_
