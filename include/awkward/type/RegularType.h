// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARTYPE_H_
#define AWKWARD_REGULARTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  /// @class RegularType
  ///
  /// @brief Describes the high level type of lists of a given length, as
  /// opposed to ListType.
  ///
  /// RegularArray nodes have this type.
  class LIBAWKWARD_EXPORT_SYMBOL RegularType: public Type {
  public:
    /// @brief Create a RegularType with a full set of parameters.
    ///
    /// @param parameters Custom parameters inherited from the Content that
    /// this type describes.
    /// @param typestr Optional string that overrides the default string
    /// representation (missing if empty).
    /// @param type The Type of the nested lists.
    /// @param size The length of each list (which is part of the type
    /// specification).
    RegularType(const util::Parameters& parameters,
                const std::string& typestr,
                const TypePtr& type,
                int64_t size);

    /// @brief The Type of the nested lists.
    const TypePtr
      type() const;

    /// @brief The length of each list (which is part of the type
    /// specification).
    int64_t
      size() const;

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
    const TypePtr type_;
    /// @brief See #size.
    const int64_t size_;
  };
}

#endif // AWKWARD_REGULARTYPE_H_
