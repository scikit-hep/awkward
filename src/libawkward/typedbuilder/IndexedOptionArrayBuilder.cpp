// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/TypedArrayBuilder.cpp", line)

#include "awkward/typedbuilder/IndexedOptionArrayBuilder.h"
#include "awkward/typedbuilder/TypedArrayBuilder.h"
#include "awkward/array/IndexedArray.h"

namespace awkward {

  ///
  IndexedOptionArrayBuilder::IndexedOptionArrayBuilder(const IndexedOptionFormPtr& form)
    : form_(form),
      form_key_(form.get()->form_key()),
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) {
    // FIXME: generate a key if this FormKey is empty
    // or already exists
    vm_output_data_ = std::string("part0-").append(*form_key_).append("-index");

    vm_func_name_ = std::string(*form_key_).append("-").append("index");

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(index_form_to_name(form_.get()->index()))
      .append("\n")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append("\n")
      .append(std::to_string(static_cast<utype>(state::null)))
      .append(" <> if").append("\n")
      .append("\n0\n")
      .append("begin\n")
      .append("pause\n\n")
      .append(content_.get()->vm_func_name()).append("\n")
      .append("variable index").append("\n")
      .append("1 index +!").append("\n")
      .append("index @ ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append("again\n")
      .append("else").append("\n\n")
      .append("variable null").append("\n")
      .append("0 null !").append("\n")
      .append("null @ ")
      .append(vm_output_data_).append(" <- stack").append("\n")
      .append("then\n")
      .append(";").append("\n");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack").append("\n");

  }

  const std::string
  IndexedOptionArrayBuilder::classname() const {
    return "IndexedOptionArrayBuilder";
  }

  const ContentPtr
  IndexedOptionArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data_);
    if (search != outputs.end()) {
       // FIXME: search->second.get()->toIndex64() length is 1 more then needed here
      Index64 index(std::static_pointer_cast<int64_t>(search->second.get()->ptr()),
                    0,
                    search->second.get()->len() - 1,
                    kernel::lib::cpu);
      return std::make_shared<IndexedOptionArray64>(Identities::none(),
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
  IndexedOptionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output() const {
    return vm_output_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func() const {
    return vm_func_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_name() const {
    return vm_func_name_;
  }

  const std::string
  IndexedOptionArrayBuilder::vm_from_stack() const {
    return vm_data_from_stack_;
  }

}
