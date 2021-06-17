// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/EmptyArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/EmptyArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  EmptyArrayBuilder::EmptyArrayBuilder(const EmptyFormPtr& form)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(LayoutBuilder::next_id()))
        : form.get()->form_key()),
      vm_empty_command_("( This does nothing. )\n"),
      vm_error_("s\" EmptyArray Builder error\"") { }

  const std::string
  EmptyArrayBuilder::classname() const {
    return "EmptyArrayBuilder";
  }

  const ContentPtr
  EmptyArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
      return std::make_shared<EmptyArray>(Identities::none(),
                                          form_.get()->parameters());
  }

  const FormPtr
  EmptyArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  EmptyArrayBuilder::vm_output() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_output_data() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func_name() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_func_type() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_from_stack() const {
    return vm_empty_command_;
  }

  const std::string
  EmptyArrayBuilder::vm_error() const {
    return vm_error_;
  }

  void
  EmptyArrayBuilder::boolean(bool x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'bool'"));
  }

  void
  EmptyArrayBuilder::int64(int64_t x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'int64'"));
  }

  void
  EmptyArrayBuilder::float64(double x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'float64'"));
  }

  void
  EmptyArrayBuilder::complex(std::complex<double> x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'complex'"));
  }

  void
  EmptyArrayBuilder::bytestring(const std::string& x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'bytestring'"));
  }

  void
  EmptyArrayBuilder::string(const std::string& x, LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'string'"));
  }

  void
  EmptyArrayBuilder::begin_list(LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'begin_list'"));
  }

  void
  EmptyArrayBuilder::end_list(LayoutBuilder* builder) {
    throw std::invalid_argument(
      std::string("EmptyArrayBuilder does not accept 'end_list'"));
  }

}
