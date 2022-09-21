// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_RECORDARRAY_H_
#define AWKWARD_RECORDARRAY_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "awkward/common.h"
#include "awkward/Identities.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class RecordForm
  ///
  /// @brief Form describing RecordArray (not a Record).
  class LIBAWKWARD_EXPORT_SYMBOL RecordForm: public Form {
  public:
    /// @brief Creates a RecordForm. See RecordArray (not Record) for
    /// documentation.
    RecordForm(bool has_identities,
               const util::Parameters& parameters,
               const FormKey& form_key,
               const util::RecordLookupPtr& recordlookup,
               const std::vector<FormPtr>& contents);

    const util::RecordLookupPtr
      recordlookup() const;

    const std::vector<FormPtr>
      contents() const;

    bool
      istuple() const override;

    const FormPtr
      content(int64_t fieldindex) const;

    const FormPtr
      content(const std::string& key) const;

    const std::vector<std::pair<std::string, FormPtr>>
      items() const;

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    void
      tojson_part(ToJson& builder, bool verbose) const override;

    const FormPtr
      shallow_copy() const override;

    const FormPtr
      with_form_key(const FormKey& form_key) const override;

    const std::string
      purelist_parameter(const std::string& key) const override;

    bool
      purelist_isregular() const override;

    int64_t
      purelist_depth() const override;

    bool
      dimension_optiontype() const override;

    const std::pair<int64_t, int64_t>
      minmax_depth() const override;

    const std::pair<bool, int64_t>
      branch_depth() const override;

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

    bool
      equal(const FormPtr& other,
            bool check_identities,
            bool check_parameters,
            bool check_form_key,
            bool compatibility_check) const override;

    const FormPtr
      getitem_field(const std::string& key) const override;

    const FormPtr
      getitem_fields(const std::vector<std::string>& keys) const override;

  private:
    const util::RecordLookupPtr recordlookup_;
    const std::vector<FormPtr> contents_;
  };

}

#endif // AWKWARD_RECORDARRAY_H_
