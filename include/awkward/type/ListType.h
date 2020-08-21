// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTTYPE_H_
#define AWKWARD_LISTTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  /// @class ListType
  ///
  /// @brief Describes the high level type of lists of variable length, as
  /// opposed to RegularType.
  ///
  /// {@link ListArrayOf ListArray} and
  /// {@link ListOffsetArrayOf ListOffsetArray} nodes have this type.
  class LIBAWKWARD_EXPORT_SYMBOL ListType: public Type {
  public:
    /// @brief Create a ListType with a full set of parameters.
    ///
    /// @param parameters Custom parameters inherited from the Content that
    /// this type describes.
    /// @param typestr Optional string that overrides the default string
    /// representation (missing if empty).
    /// @param type The Type of the nested lists.
    ListType(const util::Parameters& parameters,
             const std::string& typestr,
             const TypePtr& type);

    /// @brief The Type of the nested lists.
    const TypePtr
      type() const;

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
  };
}

#endif // AWKWARD_LISTTYPE_H_
