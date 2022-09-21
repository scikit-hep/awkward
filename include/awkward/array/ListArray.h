// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LISTARRAY_H_
#define AWKWARD_LISTARRAY_H_

#include <memory>

#include "awkward/common.h"
#include "awkward/Index.h"
#include "awkward/Identities.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class ListForm
  ///
  /// @brief Form describing ListArray.
  class LIBAWKWARD_EXPORT_SYMBOL ListForm: public Form {
  public:
    /// @brief Creates a ListForm. See {@link ListArrayOf LinkArray} for
    /// documentation.
    ListForm(bool has_identities,
             const util::Parameters& parameters,
             const FormKey& form_key,
             Index::Form starts,
             Index::Form stops,
             const FormPtr& content);

    Index::Form
      starts() const;

    Index::Form
      stops() const;

    const FormPtr
      content() const;

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
      istuple() const override;

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
    Index::Form starts_;
    Index::Form stops_;
    const FormPtr content_;
  };

}

#endif // AWKWARD_LISTARRAY_H_
