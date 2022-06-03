// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_BYTEMASKEDARRAYBUILDER_H_
#define AWKWARD_BYTEMASKEDARRAYBUILDER_H_

#include "awkward/layoutbuilder/FormBuilder.h"

namespace awkward {

  /// @class ByteMaskedArrayBuilder
  ///
  /// @brief
  template <typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL ByteMaskedArrayBuilder : public FormBuilder<T, I> {
  public:
    /// @brief Creates a ByteMaskedArrayBuilder from a full set of parameters.
    ByteMaskedArrayBuilder(const FormBuilderPtr<T, I> content,
                           const util::Parameters& parameters,
                           const std::string& form_key,
                           const std::string attribute = "mask",
                           const std::string partition = "0");

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    const std::string
      to_buffers(BuffersContainer& container, const ForthOutputBufferMap& outputs) const override;

    /// @brief AwkwardForth virtual machine instructions of the data outputs.
    const std::string
      vm_output() const override;

    /// @brief The data output key.
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
      boolean(bool x, LayoutBuilderPtr<T, I> builder) override;

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x, LayoutBuilderPtr<T, I> builder) override;

    /// @brief Adds a real value `x` to the accumulated data.
    void
      float64(double x, LayoutBuilderPtr<T, I> builder) override;

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) override;

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) override;

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x, LayoutBuilderPtr<T, I> builder) override;

    /// @brief Begins building a nested list.
    void
      begin_list(LayoutBuilderPtr<T, I> builder) override;

    /// @brief Ends a nested list.
    void
      end_list(LayoutBuilderPtr<T, I> builder) override;

    const FormBuilderPtr<T, I> content() const { return content_; }

    /// @brief String-to-JSON map that augments the meaning of this
    /// builder Form.
    ///
    /// Keys are simple strings, but values are JSON-encoded strings.
    /// For this reason, values that represent single strings are
    /// double-quoted: e.g. `"\"actual_value\""`.
    const util::Parameters&
      form_parameters() const { return parameters_; }

    ssize_t
      len(const ForthOutputBufferMap& outputs) const override {
        return content().get()->len(outputs);
      }

  private:
    /// @brief This Json Form content builder
    const FormBuilderPtr<T, I> content_;

    /// @brief This Form parameters
    const util::Parameters parameters_;

    /// @brief Forth virtual machine instructions
    /// generated from the Json Form
    ///
    /// An output buffer name is
    /// "part{partition}-{form_key}-{attribute}"
    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
    std::string vm_func_type_;
    std::string vm_data_from_stack_;
    std::string vm_error_;
  };

}

#endif // AWKWARD_BYTEMASKEDARRAYBUILDER_H_
