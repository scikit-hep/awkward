// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ByteMaskedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ByteMaskedArray.h"

namespace awkward {

  ///
  ByteMaskedArrayBuilder::ByteMaskedArrayBuilder(const ByteMaskedFormPtr& form,
                                                 const std::string attribute,
                                                 const std::string partition)
    : form_(form),
      form_key_(!form.get()->form_key() ?
        std::make_shared<std::string>(std::string("node-id")
        + std::to_string(TypedArrayBuilder::next_id()))
        : form.get()->form_key()),
      attribute_(attribute),
      partition_(partition),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);

    vm_func_type_ = content_.get()->vm_func_type();

    vm_func_.append(content_.get()->vm_func())
      .append(": ")
      .append(vm_func_name_).append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append(";").append("\n");

    vm_output_ = content_.get()->vm_output();
  }

  const std::string
  ByteMaskedArrayBuilder::classname() const {
    return "ByteMaskedArrayBuilder";
  }

  const ContentPtr
  ByteMaskedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // FIXME: how to define a mask? is it needed?
    return content_.get()->snapshot(outputs);
  }

  const FormPtr
  ByteMaskedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ByteMaskedArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_func_type() const {
    return vm_func_type_;
  }

  const std::string
  ByteMaskedArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

}
