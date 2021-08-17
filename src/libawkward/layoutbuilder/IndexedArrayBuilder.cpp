// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/IndexedArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/IndexedArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  IndexedArrayBuilder::IndexedArrayBuilder(const std::string form_key,
                                           const std::string form_index,
                                           FormBuilderPtr content,
                                           bool is_categorical,
                                           const std::string attribute,
                                           const std::string partition)
    : is_categorical_(is_categorical),
      form_index_(form_index),
      parameters_(util::Parameters()), // FIXME
      content_(content) {
    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_func_name_ = std::string(form_key).append("-")
      .append(attribute).append("-")
      .append(form_index);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(form_index)
      .append("\n")
      .append(content_.get()->vm_output())
      .append("variable index").append("\n");

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append("dup ").append(std::to_string(static_cast<utype>(state::index)))
      .append(" = if").append("\n")
      .append("drop").append("\n")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append("else").append("\n")
      .append("1 index +!").append("\n")
      .append("index @ 1- ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("then").append("\n")
      .append(";").append("\n");

    vm_error_ = content_.get()->vm_error();
  }

  const std::string
  IndexedArrayBuilder::classname() const {
    return "IndexedArrayBuilder";
  }

  const std::string
  IndexedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  IndexedArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  IndexedArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  IndexedArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  IndexedArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  IndexedArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  IndexedArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  IndexedArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<bool>(x, data)) {
        return;
      }
    }
    content_.get()->boolean(x, builder);
  }

  void
  IndexedArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<int64_t>(x, data)) {
        return;
      }
    }
    content_.get()->int64(x, builder);
  }

  void
  IndexedArrayBuilder::float64(double x, LayoutBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<double>(x, data)) {
        return;
      }
    }
    content_.get()->float64(x, builder);
  }

  void
  IndexedArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    if (is_categorical_) {
      auto const& data = content_.get()->vm_output_data();
      if (builder->find_index_of<std::complex<double>>(x, data)) {
        return;
      }
    }
    content_.get()->complex(x, builder);
  }

  void
  IndexedArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    if (is_categorical_) {
      throw std::runtime_error(
        std::string("IndexedArrayBuilder categorical 'bytestring' is not implemented yet")
        + FILENAME(__LINE__));
    }
    content_.get()->bytestring(x, builder);
  }

  void
  IndexedArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    if (is_categorical_) {
      throw std::runtime_error(
        std::string("IndexedArrayBuilder categorical 'string' is not implemented yet")
        + FILENAME(__LINE__));
    }
    content_.get()->string(x, builder);
  }

  void
  IndexedArrayBuilder::begin_list(LayoutBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  IndexedArrayBuilder::end_list(LayoutBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
