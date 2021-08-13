// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/IndexedOptionArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/IndexedOptionArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  IndexedOptionArrayBuilder::IndexedOptionArrayBuilder(const std::string form_key,
                                                       const std::string form_index,
                                                       const std::string form_content,
                                                       bool is_categorical,
                                                       const std::string attribute,
                                                       const std::string partition)
    : is_categorical_(is_categorical),
      form_index_(form_index),
      parameters_(util::Parameters()), // FIXME
      content_(LayoutBuilder::formBuilderFromJson(form_content)) {
    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_func_name_ = std::string(form_key).append("-").append(attribute);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(form_index)
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append("dup ").append(std::to_string(static_cast<utype>(state::null)))
      .append(" = if").append("\n")
      .append("drop\n")
      .append("variable null    -1 null !").append("\n")
      .append("null @ ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append("exit\n")
      .append("else\n")
      .append("variable index    1 index +!").append("\n")
      .append("index @ 1- ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("then\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");

    vm_error_ = content_.get()->vm_error();
    validate();
  }

  void
  IndexedOptionArrayBuilder::validate() const {
    if (is_categorical_) {
      throw std::invalid_argument(
        std::string("categorical form of a ") + classname()
        + std::string(" is not supported yet ")
        + FILENAME(__LINE__));
    }
  }

  const std::string
  IndexedOptionArrayBuilder::classname() const {
    return "IndexedOptionArrayBuilder";
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  IndexedOptionArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    content_.get()->boolean(x, builder);
  }

  void
  IndexedOptionArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    content_.get()->int64(x, builder);
  }

  void
  IndexedOptionArrayBuilder::float64(double x, LayoutBuilder* builder) {
    content_.get()->float64(x, builder);
  }

  void
  IndexedOptionArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    content_.get()->complex(x, builder);
  }

  void
  IndexedOptionArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    content_.get()->bytestring(x, builder);
  }

  void
  IndexedOptionArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    content_.get()->string(x, builder);
  }

  void
  IndexedOptionArrayBuilder::begin_list(LayoutBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  IndexedOptionArrayBuilder::end_list(LayoutBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
