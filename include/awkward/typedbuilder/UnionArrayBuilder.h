// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UNIONARRAYBUILDER_H_
#define AWKWARD_UNIONARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class UnionForm;
  using UnionFormPtr = std::shared_ptr<UnionForm>;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  /// @class UnionArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL UnionArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a UnionArrayBuilder from a full set of parameters.
    UnionArrayBuilder(const UnionFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOutputBufferMap& outputs) const override;

    /// @brief
    const FormPtr
      form() const override;

    /// @brief
    const std::string
      vm_output() const override;

    /// @brief
    const std::string
      vm_func() const override;

    /// @brief
    const std::string
      vm_func_name() const override;

  private:
    const UnionFormPtr form_;
    const FormKey form_key_;
    std::vector<FormBuilderPtr> contents_;

    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
  };

}

#endif // AWKWARD_UNIONARRAYBUILDER_H_
