// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/UnknownFormBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"

namespace awkward {

  ///
  UnknownFormBuilder::UnknownFormBuilder(const FormPtr& form)
    : form_(form),
      vm_empty_command_("( This does nothing. )\n"),
      vm_error_("s\"Unknown Form Builder error\"") { }

  const std::string
  UnknownFormBuilder::classname() const {
    return "UnknownFormBuilder";
  }

  const ContentPtr
  UnknownFormBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    return nullptr;
  }

  const FormPtr
  UnknownFormBuilder::form() const {
    return form_;
  }

  const std::string
  UnknownFormBuilder::vm_output() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_output_data() const {
    return vm_empty_command_;
  }
  const std::string
  UnknownFormBuilder::vm_func() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_func_name() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_func_type() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_from_stack() const {
    return vm_empty_command_;
  }

  const std::string
  UnknownFormBuilder::vm_error() const {
    return vm_error_;
  }

}
