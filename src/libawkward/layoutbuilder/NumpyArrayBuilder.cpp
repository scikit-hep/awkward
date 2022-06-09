// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/NumpyArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/NumpyArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  template <typename T, typename I>
  NumpyArrayBuilder<T, I>::NumpyArrayBuilder(const util::Parameters& parameters,
                                             const std::string& form_key,
                                             const std::string& form_primitive,
                                             const std::string& form_primitive_to_state,
                                             const std::string& form_primitive_to_vm_format,
                                             const std::string attribute,
                                             const std::string partition)
    : parameters_(parameters),
      form_key_(form_key),
      form_primitive_(form_primitive),
      is_complex_(form_primitive.rfind("complex", 0) == 0) {
    auto tmp_primitive = (is_complex_ ? "float64" : form_primitive);
    vm_error_ = std::string("s\" NumpyForm builder accepts only ")
      .append(form_primitive).append("\" ");

    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(tmp_primitive).append(" ");

    vm_func_name_ = std::string(form_key)
      .append("-")
      .append(tmp_primitive);

    vm_func_type_ = form_primitive_to_state;

    vm_func_ = std::string(": ").append(vm_func_name()).append(" ")
      .append(vm_func_type())
      .append(" = if 0 data seek data ").append(form_primitive_to_vm_format)
      .append("-> ").append(vm_output_data_)
      .append(" else ")
      .append(std::to_string(LayoutBuilder<T, I>::next_error_id()))
      .append(" err ! err @ halt then ; ");
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::classname() const {
    return "NumpyArrayBuilder";
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::to_buffers(
    BuffersContainer& container,
    const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data());
    if (search != outputs.end()) {
      container.copy_buffer(form_key() + "-data",
                            search->second.get()->ptr().get(),
                            (int64_t)((ssize_t)search->second.get()->len() * itemsize()));

      return "{\"class\": \"NumpyArray\", \"primitive\": \"" + form_primitive() + "\", "
        + this->parameters_as_string(parameters_) + " \"form_key\": \""
        + form_key() + "\"}";
    }
    throw std::invalid_argument(
      std::string("Snapshot of a ") + classname()
      + std::string(" needs data ")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_output() const {
    return vm_output_;
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_output_data() const {
    return vm_output_data_;
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_func() const {
    return vm_func_;
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_func_name() const {
    return vm_func_name_;
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_func_type() const {
    return vm_func_type_;
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_from_stack() const {
    return std::string();
  }

  template <typename T, typename I>
  const std::string
  NumpyArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    builder->add_bool(x);
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    builder->add_int64(x);
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    builder->add_double(x);
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    builder->add_complex(x);
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    builder->bytestring(x.c_str(), (int64_t)x.length());
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    builder->string(x.c_str(), (int64_t)x.length());
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
  }

  template <typename T, typename I>
  void
  NumpyArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
  }

  template class EXPORT_TEMPLATE_INST NumpyArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST NumpyArrayBuilder<int64_t, int32_t>;

}
