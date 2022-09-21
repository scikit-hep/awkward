// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_NUMPYARRAY_H_
#define AWKWARD_NUMPYARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/common.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  /// @class NumpyForm
  ///
  /// @brief Form describing NumpyArray.
  class LIBAWKWARD_EXPORT_SYMBOL NumpyForm: public Form {
  public:
    /// @brief Creates a NumpyForm. See NumpyArray for documentation.
    NumpyForm(bool has_identities,
              const util::Parameters& parameters,
              const FormKey& form_key,
              const std::vector<int64_t>& inner_shape,
              int64_t itemsize,
              const std::string& format,
              util::dtype dtype);

    const std::vector<int64_t>
      inner_shape() const;

    int64_t
      itemsize() const;

    const std::string
      format() const;

    util::dtype
      dtype() const;

    const std::string
      primitive() const;

    const TypePtr
      type(const util::TypeStrs& typestrs) const override;

    const std::string
      tostring() const override;

    const std::string
      tojson(bool pretty, bool verbose) const override;

    void
      tojson_part(ToJson& builder, bool verbose) const override;

    void
      tojson_part(ToJson& builder, bool verbose, bool toplevel) const;

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
    const std::vector<int64_t> inner_shape_;
    int64_t itemsize_;
    const std::string format_;
    const util::dtype dtype_;
  };

}

#endif // AWKWARD_NUMPYARRAY_H_
