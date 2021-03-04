// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/EmptyArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/EmptyArray.h"

namespace awkward {

  ///
  EmptyArrayBuilder::EmptyArrayBuilder(const EmptyFormPtr& form)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      vm_empty_command_("( This does nothing. )\n") { }

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

}
