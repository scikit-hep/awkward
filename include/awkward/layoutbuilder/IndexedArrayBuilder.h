// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_INDEXEDARRAYBUILDER_H_
#define AWKWARD_INDEXEDARRAYBUILDER_H_

#include "awkward/layoutbuilder/FormBuilder.h"

#include <complex>

namespace awkward {

  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  /// @class IndexedArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL IndexedArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates an IndexedArrayBuilder from a full set of parameters.
    IndexedArrayBuilder(const std::string form_key,
                        const std::string form_index,
                        const std::string form_content,
                        bool is_categorical,
                        const std::string attribute = "index",
                        const std::string partition = "0");

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

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
      boolean(bool x, LayoutBuilder* builder) override;

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x, LayoutBuilder* builder) override;

    /// @brief Adds a real value `x` to the accumulated data.
    void
      float64(double x, LayoutBuilder* builder) override;

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x, LayoutBuilder* builder) override;

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x, LayoutBuilder* builder) override;

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x, LayoutBuilder* builder) override;

    /// @brief Begins building a nested list.
    void
      begin_list(LayoutBuilder* builder) override;

    /// @brief Ends a nested list.
    void
      end_list(LayoutBuilder* builder) override;

    const FormBuilderPtr content() const { return content_; }

    const std::string&
      form_index() const { return form_index_; }

    const util::Parameters&
      form_parameters() const { return parameters_; }

  private:
    /// @brief If 'true', this array type is categorical.
    bool is_categorical_;

    const std::string form_index_;
    util::Parameters parameters_;

    /// @brief This Json Form content builder
    FormBuilderPtr content_;

    /// @brief AwkwardForth virtual machine instructions
    /// generated from the Json Form.
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

#endif // AWKWARD_INDEXEDARRAYBUILDER_H_
