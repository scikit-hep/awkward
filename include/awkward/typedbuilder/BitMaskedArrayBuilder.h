// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_BITMASKEDARRAYBUILDER_H_
#define AWKWARD_BITMASKEDARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class BitMaskedForm;
  using BitMaskedFormPtr = std::shared_ptr<BitMaskedForm>;

  /// @class BitMaskedArrayBuilder
  ///
  /// @brief BitMaskedArray builder from a BitMaskedForm
  class LIBAWKWARD_EXPORT_SYMBOL BitMaskedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a BitMaskedArrayBuilder from a full set of parameters.
    BitMaskedArrayBuilder(const BitMaskedFormPtr& form);

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

  private:
    /// @brief BitMaskedForm that defines the BitMaskedArray.
    const BitMaskedFormPtr form_;
    const FormKey form_key_;

    /// @brief Content
    FormBuilderPtr content_;

    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
  };
}

#endif // AWKWARD_BITMASKEDARRAYBUILDER_H_
