// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/EmptyArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/EmptyArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  /// @class EmptyArrayBuilder
  ///
  /// @brief EmptyArray builder from a Empty Json Form
  template <typename T, typename I>
  EmptyArrayBuilder<T, I>::EmptyArrayBuilder(const util::Parameters& parameters)
    : parameters_(parameters),
      vm_empty_command_("( This does nothing. ) "),
      vm_error_("s\" EmptyArray Builder error\"") { }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::classname() const {
    return "EmptyArrayBuilder";
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::to_buffers(BuffersContainer& container,
    const ForthOutputBufferMap& outputs) const {
    return "{\"class\": \"EmptyArray\"}";
}

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_output() const {
    return vm_empty_command_;
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_output_data() const {
    return vm_empty_command_;
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_func() const {
    return vm_empty_command_;
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_func_name() const {
    return vm_empty_command_;
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_func_type() const {
    return vm_empty_command_;
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_from_stack() const {
    return vm_empty_command_;
  }

  template <typename T, typename I>
  const std::string
  EmptyArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'bool'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'int64'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'float64'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'complex'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'bytestring'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'string'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'begin_list'"));
  }

  template <typename T, typename I>
  void
  EmptyArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'end_list'"));
  }

  template class EXPORT_TEMPLATE_INST EmptyArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST EmptyArrayBuilder<int64_t, int32_t>;

}
