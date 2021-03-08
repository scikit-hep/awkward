// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/VirtualArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/VirtualArray.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  VirtualArrayBuilder::VirtualArrayBuilder(const VirtualFormPtr& form)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
        content_(TypedArrayBuilder::formBuilderFromA(form.get()->form())) {
    vm_func_name_ = std::string(*form_key_).append("-").append("virtual");

    vm_func_.append(content_.get()->vm_func())
      .append(": ")
      .append(vm_func_name_).append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append(";").append("\n");

    vm_output_ = content_.get()->vm_output();

  }

  const std::string
  VirtualArrayBuilder::classname() const {
    return "VirtualArrayBuilder";
  }

  const ContentPtr
  VirtualArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    return content_.get()->snapshot(outputs);
  }

  const FormPtr
  VirtualArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  VirtualArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  VirtualArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  VirtualArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  VirtualArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  VirtualArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

}
