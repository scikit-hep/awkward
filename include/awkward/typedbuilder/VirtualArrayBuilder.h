// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_VIRTUALARRAYBUILDER_H_
#define AWKWARD_VIRTUALARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class VirtualForm;
  using VirtualFormPtr = std::shared_ptr<VirtualForm>;

  /// @class VirtualArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL VirtualArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a VirtualArrayBuilder from a full set of parameters.
    VirtualArrayBuilder(const VirtualFormPtr& form);

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
    const VirtualFormPtr form_;
    const FormKey form_key_;

    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
  };

}

#endif // AWKWARD_VIRTUALARRAYBUILDER_H_
