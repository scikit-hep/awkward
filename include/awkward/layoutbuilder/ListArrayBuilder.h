// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LISTARRAYBUILDER_H_
#define AWKWARD_LISTARRAYBUILDER_H_

#include "awkward/layoutbuilder/FormBuilder.h"

namespace awkward {

  /// @class ListArrayBuilder
  ///
  /// @brief
  template <typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL ListArrayBuilder : public FormBuilder<T, I> {
  public:
    /// @brief Creates a ListArrayBuilder from a full set of parameters.
    ListArrayBuilder(FormBuilderPtr<T, I> content,
                     const util::Parameters& parameters,
                     const std::string& form_key,
                     const std::string& form_starts,
                     const std::string attribute = "offsets",
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

    /// @brief true if the builder is accumulating data
    bool
      active() override;

    const
      FormBuilderPtr<T, I> content() const { return content_; }

    const std::string&
      form_starts() const { return form_starts_; }

    const util::Parameters&
      form_parameters() const { return parameters_; }

    const std::string&
      form_key() const {return form_key_; }

    ssize_t
      len(const ForthOutputBufferMap& outputs) const override {
        auto search = outputs.find(vm_output_data());
        if (search != outputs.end()) {
          return (ssize_t)search->second.get()->len();
        }
        return 0;
      }

  private:
    /// @brief This Form content builder
    FormBuilderPtr<T, I> content_;

    /// @brief This Form parameters
    const util::Parameters parameters_;

    /// @brief 'true' if this builder has received a 'begin_list' command.
    /// 'false' if the builder either has not received a 'begin_list' command
    /// or has received an 'end_list' command.
    bool begun_;

    const std::string form_starts_;
    const std::string form_key_;

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

#endif // AWKWARD_LISTARRAYBUILDER_H_
