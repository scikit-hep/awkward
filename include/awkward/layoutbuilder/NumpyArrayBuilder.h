// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_NUMPYARRAYBUILDER_H_
#define AWKWARD_NUMPYARRAYBUILDER_H_

#include "awkward/layoutbuilder/FormBuilder.h"

namespace awkward {

  /// @class NumpyArrayBuilder
  ///
  /// @brief
  template <typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL NumpyArrayBuilder : public FormBuilder<T, I> {
  public:
    /// @brief Creates a NumpyArrayBuilder from a full set of parameters.
    NumpyArrayBuilder(const util::Parameters& parameters,
                      const std::string& form_key,
                      const std::string& form_dtype,
                      const std::string& form_dtype_state,
                      const std::string& form_dtype_to_vm_format,
                      const std::string attribute = "data",
                      const std::string partition = "0");

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    const std::string
      to_buffers(BuffersContainer& container, int64_t& form_key_id, const ForthOutputBufferMap& outputs) const override;

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

    const util::Parameters&
      form_parameters() const { return parameters_; }

    const std::string&
      form_key() const {return form_key_; }

    const std::string&
      form_primitive() const {return form_primitive_; }

    ssize_t
      itemsize() const {
        return primitive_to_itemsize_.at(form_primitive());
      }

  private:

    std::map<std::string, ssize_t> primitive_to_itemsize_ {
      {"bool", 1},
      {"int8", 1},
      {"int16", 2},
      {"int32", 4},
      {"int64", 8},
      {"uint8", 1},
      {"uint16", 2},
      {"uint32", 4},
      {"uint64", 8},
      {"float16", 2},
      {"float32", 4},
      {"float64", 8},
      {"float128", 16},
      {"complex64", 8},
      {"complex128", 16},
      {"complex256", 32},
      {"datetime64", 8},
      {"timedelta64", 8},
    };

    /// @brief This Form parameters
    const util::Parameters parameters_;
    const std::string form_key_;
    const std::string form_primitive_;

    /// @brief AwkwardForth virtual machine instructions
    /// generated from the Form
    std::string vm_output_;
    std::string vm_output_data_;
    std::string vm_func_;
    std::string vm_func_name_;
    std::string vm_func_type_;
    std::string vm_data_from_stack_;
    std::string vm_error_;
  };

}

#endif // AWKWARD_NUMPYARRAYBUILDER_H_
