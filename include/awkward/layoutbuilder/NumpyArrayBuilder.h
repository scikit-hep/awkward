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
      to_buffers(BuffersContainer& container, const ForthOutputBufferMap& outputs) const override;

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
        if (form_primitive() == "float64") {
          return sizeof(double);
        } else if (form_primitive() == "int64") {
          return sizeof(int64_t);
        } else if (form_primitive() == "complex128") {
          return sizeof(std::complex<double>);
        } else if (form_primitive() == "bool") {
          return sizeof(bool);
        }
        return util::dtype_to_itemsize(util::name_to_dtype(form_primitive()));
      }

    ssize_t
      len(const ForthOutputBufferMap& outputs) const override {
        auto search = outputs.find(vm_output_data());
        if (search != outputs.end()) {
          return (is_complex() ?
            (ssize_t)search->second.get()->len() >> 1 :
            (ssize_t)search->second.get()->len());
        }
        return 0;
      }

    bool
      is_complex() const override {
        return is_complex_;
    }

  private:

    /// @brief This Form parameters
    const util::Parameters parameters_;
    const std::string form_key_;
    const std::string form_primitive_;
    bool is_complex_;

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
