// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/NumpyArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/NumpyArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  NumpyArrayBuilder::NumpyArrayBuilder(const std::string form_key,
                                       const std::string form_primitive,
                                       const std::string form_primitive_to_state,
                                       const std::string form_primitive_to_vm_format,
                                       const std::string attribute,
                                       const std::string partition)
    : parameters_(util::Parameters()), // FIXME
      form_primitive_(form_primitive) {
    vm_error_ = std::string("s\" NumpyForm builder accepts only ")
      .append(form_primitive).append("\"\n");

    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(form_primitive).append("\n");

    vm_func_name_ = std::string(form_key)
      .append("-")
      .append(form_primitive);

    vm_func_type_ = form_primitive_to_state;

    vm_func_ = std::string(": ").append(vm_func_name()).append("\n")
      .append(vm_func_type())
      .append(" = if").append("\n")
      .append("0 data seek").append("\n")
      .append("data ").append(form_primitive_to_vm_format)
      .append("-> ").append(vm_output_data_).append("\n")
      .append("else").append("\n")
      .append(std::to_string(LayoutBuilder::next_error_id())).append(" err ! err @ halt").append("\n")
      .append("then").append("\n")
      .append(";").append("\n");
  }

  const std::string
  NumpyArrayBuilder::classname() const {
    return "NumpyArrayBuilder";
  }

  const std::string
  NumpyArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  NumpyArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  NumpyArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  NumpyArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  NumpyArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  NumpyArrayBuilder::vm_from_stack() const {
    return std::string();
  }

  const std::string
  NumpyArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  NumpyArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    builder->add<bool>(x);
  }

  void
  NumpyArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    builder->add<int64_t>(x);
  }

  void
  NumpyArrayBuilder::float64(double x, LayoutBuilder* builder) {
    builder->add<double>(x);
  }

  void
  NumpyArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    builder->add<std::complex<double>>(x);
  }

  void
  NumpyArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    builder->bytestring(x.c_str(), (int64_t)x.length());
  }

  void
  NumpyArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    builder->string(x.c_str(), (int64_t)x.length());
  }

  void
  NumpyArrayBuilder::begin_list(LayoutBuilder* builder) {
  }

  void
  NumpyArrayBuilder::end_list(LayoutBuilder* builder) {
  }

}
