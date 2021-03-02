// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UNMASKEDARRAYBUILDER_H_
#define AWKWARD_UNMASKEDARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class UnmaskedForm;
  using UnmaskedFormPtr = std::shared_ptr<UnmaskedForm>;

  /// @class UnmaskedArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL UnmaskedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an UnmaskedArrayBuilder from a full set of parameters.
    UnmaskedArrayBuilder(const UnmaskedFormPtr& form);

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
    const UnmaskedFormPtr form_;
    const FormKey form_key_;

    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
  };

}

#endif // AWKWARD_UNMASKEDARRAYBUILDER_H_
