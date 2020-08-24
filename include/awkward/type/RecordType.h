// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDTYPE_H_
#define AWKWARD_RECORDTYPE_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "awkward/util.h"

#include "awkward/type/Type.h"

namespace awkward {
  /// @class RecordType
  ///
  /// @brief Describes the high level type of data containing tuples or
  /// records.
  ///
  /// RecordArray nodes have this type.
  class LIBAWKWARD_EXPORT_SYMBOL RecordType: public Type {
  public:
    /// @brief Create an RecordArray with a full set of parameters.
    ///
    /// @param parameters Custom parameters inherited from the Content that
    /// this type describes.
    /// @param typestr Optional string that overrides the default string
    /// representation (missing if empty).
    /// @param types The Type of each field (in order).
    /// @param recordlookup A `std::shared_ptr<std::vector<std::string>>`
    /// optional list of key names.
    /// If absent (`nullptr`), the data are tuples; otherwise, they are
    /// records. The number of names must match the number of #types.
    RecordType(const util::Parameters& parameters,
               const std::string& typestr,
               const std::vector<TypePtr>& types,
               const util::RecordLookupPtr& recordlookup);

    /// @brief Creates a RecordArray without a #recordlookup (set it to
    /// `nullptr`).
    ///
    /// See #RecordType for a full list of parameters.
    RecordType(const util::Parameters& parameters,
               const std::string& typestr,
               const std::vector<TypePtr>& types);

    /// @brief The Type of each field (in order).
    const std::vector<TypePtr>
      types() const;

    /// @brief A `std::shared_ptr<std::vector<std::string>>`
    /// optional list of key names.
    ///
    /// If absent (`nullptr`), the data are tuples; otherwise, they are
    /// records. The number of names must match the number of #types.
    const util::RecordLookupPtr
      recordlookup() const;

    /// @brief Returns `true` if #recordlookup is `nullptr`; `false` otherwise.
    bool
      istuple() const;

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

    /// @brief Returns the field at a given index.
    ///
    /// Equivalent to `types[fieldindex]`.
    const TypePtr
      field(int64_t fieldindex) const;

    /// @brief Returns the field with a given key name.
    ///
    /// Equivalent to `types[fieldindex(key)]`.
    const TypePtr
      field(const std::string& key) const;

    /// @brief Returns all the fields.
    ///
    /// Equivalent to `types`.
    const std::vector<TypePtr>
      fields() const;

    /// @brief Returns key, field pairs for all fields.
    const std::vector<std::pair<std::string, TypePtr>>
      fielditems() const;

    /// @brief Returns this RecordType without #recordlookup, converting any
    /// records into tuples.
    const TypePtr
      astuple() const;

  private:
    /// @brief See #types.
    const std::vector<TypePtr> types_;
    /// @brief See #recordlookup.
    const util::RecordLookupPtr recordlookup_;
  };
}

#endif // AWKWARD_RECORDTYPE_H_
