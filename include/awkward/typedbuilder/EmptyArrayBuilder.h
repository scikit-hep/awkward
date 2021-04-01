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

  private:
    const EmptyFormPtr form_;
    const FormKey form_key_;

    std::string vm_empty_command_;
    std::string vm_error_;
  };

}

#endif // AWKWARD_EMPTYARRAYBUILDER_H_
