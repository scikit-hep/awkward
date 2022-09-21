// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_BYTEMASKEDARRAY_H_
#define AWKWARD_BYTEMASKEDARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Index.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class ByteMaskedForm
  ///
  /// @brief Form describing ByteMaskedArray.
  class LIBAWKWARD_EXPORT_SYMBOL ByteMaskedForm: public Form {
  public:
    /// @brief Creates a ByteMaskedForm. See ByteMaskedArray for documentation.
    ByteMaskedForm(bool has_identities,
                   const util::Parameters& parameters,
                   const FormKey& form_key,
                   Index::Form mask,
                   const FormPtr& content,
                   bool valid_when);

    Index::Form
      mask() const;

    const FormPtr
      content() const;

    bool
      valid_when() const;

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

    const FormPtr
      simplify_optiontype() const;

  private:
    Index::Form mask_;
    const FormPtr content_;
    bool valid_when_;
  };

}

#endif // AWKWARD_BYTEMASKEDARRAY_H_
