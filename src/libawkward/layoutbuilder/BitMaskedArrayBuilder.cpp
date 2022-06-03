// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/BitMaskedArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/BitMaskedArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  /// @bclass BitMaskedArrayBuilder
  ///
  /// @brief BitMaskedArray builder from a Bit Masked Json Form
  template <typename T, typename I>
  BitMaskedArrayBuilder<T, I>::BitMaskedArrayBuilder(FormBuilderPtr<T, I> content,
                                                     const util::Parameters& parameters,
                                                     const std::string& form_key,
                                                     const std::string attribute,
                                                     const std::string partition)
    : content_(content),
      parameters_(parameters) {
    vm_func_name_ = std::string(form_key).append("-").append(attribute);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_func_.append(content_.get()->vm_func())
      .append(": ")
      .append(vm_func_name_).append(" ")
      .append(content_.get()->vm_func_name())
      .append(" ; ");

    vm_output_ = content_.get()->vm_output();
    vm_error_ = content_.get()->vm_error();
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::classname() const {
    return "BitMaskedArrayBuilder";
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::to_buffers(BuffersContainer& container, const ForthOutputBufferMap& outputs) const {
    return content().get()->to_buffers(container, outputs);
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_output() const {
    return vm_output_;
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_output_data() const {
    return vm_output_data_;
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_func() const {
    return vm_func_;
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_func_name() const {
    return vm_func_name_;
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_func_type() const {
    return vm_func_type_;
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  template <typename T, typename I>
  const std::string
  BitMaskedArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->boolean(x, builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->int64(x, builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->float64(x, builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->complex(x, builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->bytestring(x, builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->string(x, builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
    content_.get()->begin_list(builder);
  }

  template <typename T, typename I>
  void
  BitMaskedArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
    content_.get()->end_list(builder);
  }

  template class EXPORT_TEMPLATE_INST BitMaskedArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST BitMaskedArrayBuilder<int64_t, int32_t>;

}
