// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/RegularArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/RegularArray.h"

namespace awkward {

  ///
  RegularArrayBuilder::RegularArrayBuilder(const RegularFormPtr& form,
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
    vm_output_data_ = std::string("part")
      .append(partition_).append("-")
      .append(*form_key_).append("-")
      .append(attribute_);

    vm_output_ = content_.get()->vm_output();

    vm_func_name_ = std::string(*form_key_).append("-").append(attribute_);

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append(";").append("\n");
  }

  const std::string
  RegularArrayBuilder::classname() const {
    return "RegularArrayBuilder";
  }

  const ContentPtr
  RegularArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    ContentPtr out;
    if(content_ != nullptr) {
      out = std::make_shared<RegularArray>(Identities::none(),
                                           form_.get()->parameters(),
                                           content_.get()->snapshot(outputs),
                                           form_.get()->size());
    }
    return out;
  }

  const FormPtr
  RegularArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  RegularArrayBuilder::vm_output() const {
    return vm_output_;
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

}
