// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_EMPTYARRAYBUILDER_H_
#define AWKWARD_EMPTYARRAYBUILDER_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/forth/ForthMachine.h"
#include "awkward/typedbuilder/FormBuilder.h"

#include <complex>

namespace awkward {

  class EmptyForm;
  using EmptyFormPtr = std::shared_ptr<EmptyForm>;

  /// @class EmptyArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL EmptyArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an EmptyArrayBuilder from a full set of parameters.
    EmptyArrayBuilder(const EmptyFormPtr& form);

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOutputBufferMap& outputs) const override;

    /// @brief The Form describing the array.
    const FormPtr
      form() const override;

    /// @brief AwkwardForth virtual machine instructions of the data outputs.
    const std::string
      vm_output() const override;

    /// @brief AwkwardForth virtual machine data output key.
    const std::string
      vm_output_data() const override;

    /// @brief AwkwardForth virtual machine instructions of the array builder function.
    const std::string
      vm_func() const override;

    /// @brief The array builder VM function name.
    const std::string
      vm_func_name() const override;

    /// @brief The array builder VM function type.
    const std::string
      vm_func_type() const override;

    /// @brief AwkwardForth virtual machine instructions to retrieve the data from
    /// the VM stack.
    const std::string
      vm_from_stack() const override;

    /// @brief Error messages in the AwkwardForth virtual machine instructions.
    const std::string
      vm_error() const override;

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x, TypedArrayBuilder* builder) override;

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x, TypedArrayBuilder* builder) override;

    /// @brief Adds a real value `x` to the accumulated data.
    void
      float64(double x, TypedArrayBuilder* builder) override;

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x, TypedArrayBuilder* builder) override;

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x, TypedArrayBuilder* builder) override;

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x, TypedArrayBuilder* builder) override;

    /// @brief Begins building a nested list.
    void
      begin_list(TypedArrayBuilder* builder) override;

    /// @brief Ends a nested list.
    void
      end_list(TypedArrayBuilder* builder) override;

  private:
    /// @brief EmptyForm that defines the EmptyArray.
    const EmptyFormPtr form_;
    /// @brief an output buffer name is
    /// "part{partition}-{form_key}-{attribute}"
    const FormKey form_key_;

    /// @brief An empty command.
    std::string vm_empty_command_;
    /// @brief An error message.
    std::string vm_error_;
  };

}

#endif // AWKWARD_EMPTYARRAYBUILDER_H_
