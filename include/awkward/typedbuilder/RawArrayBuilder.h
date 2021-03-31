// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_RAWARRAYBUILDER_H_
#define AWKWARD_RAWARRAYBUILDER_H_

#include "awkward/typedbuilder/FormBuilder.h"

namespace awkward {

  class RawForm;
  using RawFormPtr = std::shared_ptr<RawForm>;

  /// @class RawArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL RawArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a RawArrayBuilder from a full set of parameters.
    RawArrayBuilder(const RawFormPtr& form);

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
      vm_output_data() const override;

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

    /// @brief
    const std::string
      vm_error() const override;

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x, TypedArrayBuilder* builder) override;

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x, TypedArrayBuilder* builder) override;

  private:
    const RawFormPtr form_;
    const FormKey form_key_;

    /// @brief Forth virtual machine instructions
    /// generated from the Form
    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
    std::string vm_func_type_;
    std::string vm_data_from_stack_;
    std::string vm_error_;
  };

}

#endif // AWKWARD_RAWARRAYBUILDER_H_
