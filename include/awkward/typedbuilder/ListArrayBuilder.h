// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LISTARRAYBUILDER_H_
#define AWKWARD_LISTARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class ListForm;
  using ListFormPtr = std::shared_ptr<ListForm>;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  /// @class ListArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL ListArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a ListArrayBuilder from a full set of parameters.
    ListArrayBuilder(const ListFormPtr& form,
                     const std::string attribute = "offsets",
                     const std::string partition = "0");

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

    /// @brief
    const std::string
      vm_func_type() const override;

    /// @brief
    const std::string
      vm_from_stack() const override;

  private:
    /// @brief This builder Form
    const ListFormPtr form_;

    /// @brief an output buffer name is
    /// "part{partition}-{form_key}-{attribute}"
    const FormKey form_key_;
    const std::string attribute_;
    const std::string partition_;

    /// @brief This Form content builder
    FormBuilderPtr content_;

    /// @brief Forth virtual machine instructions
    /// generated from the Form
    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
    std::string vm_func_type_;
    std::string vm_data_from_stack_;
  };

}

#endif // AWKWARD_LISTARRAYBUILDER_H_
