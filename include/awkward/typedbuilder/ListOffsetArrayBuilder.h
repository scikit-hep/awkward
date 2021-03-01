// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LISTOFFSETARRAYBUILDER_H_
#define AWKWARD_LISTOFFSETARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class ListOffsetForm;
  using ListOffsetFormPtr = std::shared_ptr<ListOffsetForm>;

  /// @class ListOffsetArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL ListOffsetArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a ListOffsetArrayBuilder from a full set of parameters.
    ListOffsetArrayBuilder(const ListOffsetFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOtputBufferMap& outputs) const override;

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
      vm_from_stack() const override;

  private:
    const ListOffsetFormPtr form_;
    const FormKey form_key_;
    FormBuilderPtr content_;

    std::string vm_output_;
    std::string vm_output_data_;
    std::string vm_func_;
    std::string vm_func_name_;
    std::string vm_data_from_stack_;
  };

}

#endif // AWKWARD_LISTOFFSETARRAYBUILDER_H_
