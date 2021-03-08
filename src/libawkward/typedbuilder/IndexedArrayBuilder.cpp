// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/IndexedArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/IndexedArray.h"

namespace awkward {

  ///
  IndexedArrayBuilder::IndexedArrayBuilder(const IndexedFormPtr& form,
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

    vm_func_name_ = std::string(*form_key_).append("-")
      .append(attribute_).append("-")
      .append(index_form_to_name(form_.get()->index()));

    vm_func_type_ = content_.get()->vm_func_type();

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->index()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append("variable index    1 index +!").append("\n")
      .append("index @ ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");
  }

  const std::string
  IndexedArrayBuilder::classname() const {
    return "IndexedArrayBuilder";
  }

  const ContentPtr
  IndexedArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
       // FIXME: search->second.get()->toIndex64() length is 1 more then needed here
      Index64 index(std::static_pointer_cast<int64_t>(search->second.get()->ptr()),
                    0,
                    search->second.get()->len() - 1,
                    kernel::lib::cpu);
      return std::make_shared<IndexedArray64>(Identities::none(),
                                              form_.get()->parameters(),
                                              index,
                                              content_.get()->snapshot(outputs));
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs an index ")
        + FILENAME(__LINE__));
  }

  const FormPtr
  IndexedArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedArrayBuilder::vm_output() const {
    return vm_output_;
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

}
