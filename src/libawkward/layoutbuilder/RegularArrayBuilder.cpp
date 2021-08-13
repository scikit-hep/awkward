// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/RegularArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/RegularArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  RegularArrayBuilder::RegularArrayBuilder(const std::string form_key,
                                           const std::string form_content,
                                           const std::string attribute,
                                           const std::string partition)
    : parameters_(util::Parameters()), // FIXME
      form_size_(1), // FIXME
      content_(LayoutBuilder::formBuilderFromJson(form_content)) {
    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_output_ = content_.get()->vm_output();

    vm_func_name_ = std::string(form_key).append("-").append(attribute);

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append(";").append("\n");

    vm_error_.append(content_.get()->vm_error());
  }

  const std::string
  RegularArrayBuilder::classname() const {
    return "RegularArrayBuilder";
  }

  const std::string
  RegularArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  RegularArrayBuilder::vm_output_data() const {
    return vm_output_data_;
  }

  const std::string
  RegularArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  RegularArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  RegularArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  RegularArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  const std::string
  RegularArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  RegularArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    content_.get()->boolean(x, builder);
  }

  void
  RegularArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    content_.get()->int64(x, builder);
  }

  void
  RegularArrayBuilder::float64(double x, LayoutBuilder* builder) {
    content_.get()->float64(x, builder);
  }

  void
  RegularArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    content_.get()->complex(x, builder);
  }

  void
  RegularArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    content_.get()->bytestring(x, builder);
  }

  void
  RegularArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    content_.get()->string(x, builder);
  }

  void
  RegularArrayBuilder::begin_list(LayoutBuilder* builder) {
    content_.get()->begin_list(builder);
  }

  void
  RegularArrayBuilder::end_list(LayoutBuilder* builder) {
    content_.get()->end_list(builder);
  }

}
