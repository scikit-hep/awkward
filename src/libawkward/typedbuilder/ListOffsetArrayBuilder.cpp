// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/ListOffsetArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/ListOffsetArray.h"

namespace awkward {

  ///
  ListOffsetArrayBuilder::ListOffsetArrayBuilder(const ListOffsetFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-offsets");

    vm_func_name_ = std::string(*form_key_).append("-").append("list");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->offsets()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(std::to_string(static_cast<utype>(state::begin_list)))
      .append(" <> if").append("\n")
      .append("halt").append("\n")
      .append("then").append("\n")
      .append("\n")
      .append("0").append("\n")
      .append("begin").append("\n")
      .append("pause").append("\n")
      .append("dup ")
      .append(std::to_string(static_cast<utype>(state::end_list)))
      .append(" = if").append("\n")
      .append("drop").append("\n")
      .append(vm_output_data_).append(" +<- stack").append("\n")
      .append("exit").append("\n")
      .append("else").append("\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("1+").append("\n")
      .append("then").append("\n")
      .append("again").append("\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");
  }

  const std::string
  ListOffsetArrayBuilder::classname() const {
    return "ListOffsetArrayBuilder";
  }

  const ContentPtr
  ListOffsetArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
      return std::make_shared<ListOffsetArray64>(Identities::none(),
                                                 form_.get()->parameters(),
                                                 search->second.get()->toIndex64(),
                                                 content_.get()->snapshot(outputs));
    }
    throw std::invalid_argument(
        std::string("Snapshot of a ") + classname()
        + std::string(" needs offsets")
        + FILENAME(__LINE__));
  }

  const FormPtr
  ListOffsetArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  ListOffsetArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  ListOffsetArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

}
