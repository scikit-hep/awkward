// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UNIONARRAYBUILDER_H_
#define AWKWARD_UNIONARRAYBUILDER_H_

#include "awkward/layoutbuilder/FormBuilder.h"

namespace awkward {

  /// @class UnionArrayBuilder
  ///
  /// @brief
  template <typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL UnionArrayBuilder : public FormBuilder<T, I> {
  public:
    /// @brief Creates a UnionArrayBuilder from a full set of parameters.
    UnionArrayBuilder(const std::vector<FormBuilderPtr<T, I>>& contents,
                      const util::Parameters& parameters,
                      const std::string& form_key,
                      const std::string& form_tags,
                      const std::string& form_index,
                      const std::string attribute = "union",
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

    /// @brief Sets the tag for next input data.
    void
      tag(int8_t x) override;

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

    const std::vector<FormBuilderPtr<T, I>>&
      contents() const { return contents_; }

    const std::string&
      vm_output_tags() const { return vm_output_tags_; }

    const std::string&
      form_index() const { return form_index_; }

    const util::Parameters&
      form_parameters() const { return parameters_; }

    const std::string&
      form_key() const {return form_key_; }

    ssize_t
      len(const ForthOutputBufferMap& outputs) const override {
        auto search = outputs.find(vm_output_tags());
        if (search != outputs.end()) {
          return (ssize_t)search->second.get()->len();
        }
        return 0;
      }

  private:
    /// @brief This Form content builders
    std::vector<FormBuilderPtr<T, I>> contents_;

    /// @brief This Form parameters
    const util::Parameters parameters_;
    const std::string form_key_;

    /// @brief UnionArray tag
    int8_t tag_;

    const std::string form_index_;

    /// @brief Forth virtual machine instructions
    /// generated from the Form
    std::string vm_output_data_;
    std::string vm_output_;
    std::string vm_func_name_;
    std::string vm_func_;
    std::string vm_func_type_;
    std::string vm_data_from_stack_;
    std::string vm_output_index_;
    std::string vm_output_tags_;
    std::string vm_error_;
  };

}

#endif // AWKWARD_UNIONARRAYBUILDER_H_
