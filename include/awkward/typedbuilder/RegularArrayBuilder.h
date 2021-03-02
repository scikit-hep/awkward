// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_REGULARARRAYBUILDER_H_
#define AWKWARD_REGULARARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class RegularForm;
  using RegularFormPtr = std::shared_ptr<RegularForm>;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  /// @class RegularArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL RegularArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a RegularArrayBuilder from a full set of parameters.
    RegularArrayBuilder(const RegularFormPtr& form);

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
    const RegularFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;

    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
  };

}

#endif // AWKWARD_REGULARARRAYBUILDER_H_
