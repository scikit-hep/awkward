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
      content_(TypedArrayBuilder::formBuilderFromA(form.get()->content())) { }

  const std::string
  IndexedOptionArrayBuilder::classname() const {
    return "IndexedOptionArrayBuilder";
  }

  const ContentPtr
  IndexedOptionArrayBuilder::snapshot(const ForthOutputBufferMap& outputs) const {
    // if(content_ != nullptr) {
    //   Index64 index(reinterpret_pointer_cast<int64_t>(data_), 0, length_, kernel::lib::cpu);
    //   return std::make_shared<IndexedOptionArray64>(Identities::none(),
    //                                                 form_.get()->parameters(),
    //                                                 index,
    //                                                 content_.get()->snapshot(outputs));
    // }
    // else {
    //   throw std::invalid_argument(
    //     std::string("Form of a ") + classname()
    //     + std::string(" needs another Form as its content")
    //     + FILENAME(__LINE__));
    // }
    return nullptr;
  }

  const FormPtr
  IndexedOptionArrayBuilder::form() const {
    return std::static_pointer_cast<Form>(form_);
  }

  const std::string
  IndexedOptionArrayBuilder::vm_output() const {
    return std::string("\n");
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func() const {
    return std::string(": ")
      .append(vm_func_name())
      .append(";\n");
  }

  const std::string
  IndexedOptionArrayBuilder::vm_func_name() const {
    return std::string(*form_key_)
      .append("-")
      .append("index");
  }

}
